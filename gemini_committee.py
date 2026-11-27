#!/usr/bin/env python3
"""Gemini-assisted premarket risk review for the US equity opportunity scan.

This is one Gemini model prompted to examine five distinct review perspectives;
it is not five independent models or an independently validated forecast.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import requests

OUT = Path("output")
MODEL = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.8-flash"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


def read_csv(path: Path, limit: int = 30, columns: list[str] | None = None) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if columns:
        rows = [{k: row.get(k, "") for k in columns if k in row} for row in rows]
    return rows[:limit]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def build_packet() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_6m_candidates": read_csv(
            OUT / "top_25_6m.csv", 25,
            ["rank", "ticker", "as_of", "prob_up", "expected_return_model_estimate",
             "opportunity_score", "volatility_6m_annualized", "drawdown_6m", "signal",
             "combined_news_score", "news_confidence", "market_news_count_7d"]),
        "validation": read_csv(OUT / "validation_report.csv", 10),
        "factor_validation": read_csv(OUT / "factor_validation_report.csv", 10),
        "scanner_run_summary": read_csv(OUT / "run_summary.csv", 3),
        "market_risk_summary": read_json(OUT / "run_summary_v4.json"),
        "ai_infrastructure_risk": read_csv(OUT / "ai_infrastructure_risk_summary_v4.csv", 5),
        "ai_infrastructure_risk_details": read_csv(OUT / "ai_infrastructure_risk_details_v4.csv", 12),
        "nq_soxl_forecasts": read_csv(OUT / "market_forecasts_v4.csv", 20),
        "recent_ai_news": read_csv(OUT / "ai_news_v4.csv", 35),
        "market_dashboard_excerpt": (
            (OUT / "market_dashboard_v4.md").read_text(encoding="utf-8")[:18000]
            if (OUT / "market_dashboard_v4.md").exists() else ""
        ),
    }


def validate_packet(packet: dict) -> None:
    if not packet["top_6m_candidates"]:
        raise RuntimeError("Missing output/top_25_6m.csv or it contains no candidate rows.")
    if not packet["market_risk_summary"]:
        raise RuntimeError("Missing output/run_summary_v4.json; market risk model did not produce its summary.")


def make_prompt(packet: dict) -> str:
    return """你是美股盘前风险审查委员会。请只根据下方数据作判断，不得补造价格、新闻、财报、期权仓位或资金流数据。
请用中文输出，明确区分“数据事实 / 推断 / 未知”。数据缺失必须写“未提供”，不能自行填补。

请模拟以下五个审查席位分别提出意见，然后形成委员会结论：
1. 量化筛选席：审查候选股排名、上涨概率、预期收益估计、样本量和验证指标；指出模型是否缺乏可证明优势。
2. 市场环境席：审查 NQ/QQQ/SOXL 相关风险、市场压力指标和宏观背景；仅引用数据包中已有指标。
3. 新闻与基本面席：审查新闻标题、AI 公司财务和新闻可信度；标题不等于事实，来源不明或重复新闻要降权。
4. AI 资本开支/融资风险席：审查资本开支、现金流、债务与利息覆盖相关信息；缺失数据不能视为安全。
5. 风控席：识别高波动、深回撤、拥挤/相关性、事件风险；给出哪些股票应排除、观察或保留。

最终必须包含：
A. 盘前总判断：偏多 / 中性 / 偏空，以及主要依据。
B. 红色排雷清单：Ticker、风险、证据、是否剔除。
C. 通过初筛的观察名单：最多10只，按优先级排序；每只写入选理由、最大未知项。
D. 委员会分歧：哪些判断存在证据冲突。
E. 今日执行纪律：哪些情况下不应追高或开仓；不要给出虚构的精确入场价、止损价或胜率。
F. 数据完整性：明确列出缺失文件、缺失字段和数据局限。

注意：
- 模型的 prob_up 是历史训练模型输出，不是经校准的真实胜率。
- expected_return_model_estimate 是模型估计，不是保证收益。
- 历史验证若显示 AUC 接近 0.5、IC 接近 0 或超额收益不稳定，应明确指出没有证明预测优势。
- 本报告是研究筛查，不构成投资建议。
- 你是一个 Gemini 模型扮演五个审查视角，不得声称这是五个独立 AI 模型投票。

数据包 JSON：
""" + json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY secret is missing or empty.")
    url = f"{API_ROOT}/{MODEL}:generateContent"
    response = requests.post(
        url,
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 7000},
        },
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        timeout=120,
    )
    if response.status_code != 200:
        # Do not print request headers or credentials in logs.
        detail = response.text[:1800].replace(api_key, "[REDACTED]")
        raise RuntimeError(f"Gemini API returned HTTP {response.status_code}: {detail}")
    payload = response.json()
    parts = (
        payload.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
    if not text.strip():
        feedback = payload.get("promptFeedback", {})
        raise RuntimeError(f"Gemini returned no text content. promptFeedback={feedback}")
    return text.strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    packet = build_packet()
    validate_packet(packet)
    result = call_gemini(make_prompt(packet))
    report = (
        "# 美股盘前量化初筛 · Gemini 迷你委员会排雷\n\n"
        f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}\n"
        f"- Gemini 模型：{MODEL}\n"
        "- 说明：单个 Gemini 模型按五个审查视角进行结构化审查，不是五个独立模型投票。\n\n"
        + result + "\n"
    )
    (OUT / "gemini_committee_premarket.md").write_text(report, encoding="utf-8")
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "status": "success",
        "candidate_count": len(packet["top_6m_candidates"]),
        "news_item_count": len(packet["recent_ai_news"]),
        "output": "output/gemini_committee_premarket.md",
        "method_note": "Single Gemini model using five role-based review perspectives; not an independent ensemble.",
    }
    (OUT / "gemini_committee_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
