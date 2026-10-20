"""Configurable FININ equation-level reproduction, not recovered author code.

Frozen language-model vectors are inputs. Eqs. 1--5 share the text projection
and use separate market/news numerical and concatenation encoders; Eq. 6 uses
news-only query/key attention; Eq. 7 concatenates complete daily histories.

The paper does not fully specify MLP activations/depth, residual connections or
normalization. Width 32, two-layer ReLU MLPs and ``plain`` self-attention are
declared implementation choices. ``residual_norm``, missing-day zeros,
``news_only`` and optional prior-bias initialization are explicit extensions.
No positional encoding is used within a day's unordered news set.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _mlp(input_width: int, output_width: int, width: int, layers: int, dropout: float) -> nn.Sequential:
    modules: list[nn.Module] = []
    for _ in range(layers - 1):
        modules.extend((nn.Linear(input_width, width), nn.ReLU(), nn.Dropout(dropout)))
        input_width = width
    modules.append(nn.Linear(input_width, output_width))
    return nn.Sequential(*modules)


class PaperFININ(nn.Module):
    """Daily independent news interaction followed by a history-aware predictor.

    Batch keys follow ReducedFININ: headline_vectors [B,T,K,E], sentiment
    [B,T,K,3], market_features [B,T,6], mask [B,T,K] (boolean), and
    market_description [E], [B,E] or [B,T,E]. T=1 permits legacy shapes.
    ``news_only`` removes actual price features both from query conditioning and
    the direct predictor branch; the fixed description can still define a query.
    Padded headlines are sanitized before encoders and receive exactly zero
    weights. A day with no headlines has a zero news representation.
    """

    def __init__(
        self, variant: str = "full", *, text_width: int = 384, width: int = 32,
        attention_width: int = 32, attention_heads: int = 1, history_days: int = 1,
        mlp_layers: int = 2, dropout: float = 0.1,
        attention_style: str = "plain", initial_probability: float | None = None,
    ):
        super().__init__()
        if variant not in {"full", "mean_pool", "no_self_attention", "news_only"}:
            raise ValueError(f"Unknown variant: {variant}")
        if attention_style not in {"plain", "residual_norm"}:
            raise ValueError(f"Unknown attention style: {attention_style}")
        if min(text_width, width, attention_width, attention_heads, history_days, mlp_layers) < 1:
            raise ValueError("All dimensions and MLP depth must be positive")
        if attention_width % attention_heads:
            raise ValueError("attention_width must divide evenly among attention_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if initial_probability is not None and not 0 < initial_probability < 1:
            raise ValueError("initial_probability must be strictly between zero and one")
        self.variant, self.attention_style = variant, attention_style
        self.text_width, self.width = text_width, width
        self.attention_width, self.attention_heads = attention_width, attention_heads
        self.history_days = history_days
        self.text_projection = nn.Linear(text_width, width)
        self.market_numbers = _mlp(6, width, width, mlp_layers, dropout)
        self.news_numbers = _mlp(3, width, width, mlp_layers, dropout)
        self.market_fusion = _mlp(2 * width, width, width, mlp_layers, dropout)
        self.news_fusion = _mlp(2 * width, width, width, mlp_layers, dropout)
        self.self_query = nn.Linear(width, attention_width)
        self.self_key = nn.Linear(width, attention_width)
        self.self_value = nn.Linear(width, attention_width)
        self.self_output = nn.Linear(attention_width, width)
        self.self_dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(width) if attention_style == "residual_norm" else nn.Identity()
        self.market_query = nn.Linear(width, attention_width)
        self.news_key = nn.Linear(width, attention_width)
        self.predictor = _mlp(2 * width * history_days, 1, width, mlp_layers, dropout)
        if initial_probability is not None:
            # Declared extension: a zero output head starts at training prevalence.
            nn.init.zeros_(self.predictor[-1].weight)
            nn.init.constant_(self.predictor[-1].bias, math.log(initial_probability / (1 - initial_probability)))

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # All-empty rows remain finite and exactly zero instead of NaN.
        weights = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(dim=-1)
        weights = weights.masked_fill(~mask, 0)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)

    def _inputs(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        headlines, sentiment = batch["headline_vectors"], batch["sentiment"]
        market, mask, description = batch["market_features"], batch["mask"], batch["market_description"]
        legacy = headlines.ndim == 3
        if legacy:
            headlines, sentiment = headlines[:, None], sentiment[:, None]
            market, mask = market[:, None], mask[:, None]
        if headlines.ndim != 4:
            raise ValueError("headline_vectors must be [B,T,K,E] or [B,K,E]")
        b, t, k, e = headlines.shape
        if t != self.history_days or e != self.text_width or k < 1:
            raise ValueError("Headline geometry disagrees with configured history/text width or has no slots")
        if sentiment.shape != (b, t, k, 3) or market.shape != (b, t, 6) or mask.shape != (b, t, k):
            raise ValueError("Sentiment, market or mask geometry disagrees with headline geometry")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be boolean")
        if description.shape == (e,):
            description = description[None, None].expand(b, t, e)
        elif description.shape == (b, e):
            description = description[:, None].expand(b, t, e)
        elif description.shape != (b, t, e):
            raise ValueError("market_description must be [E], [B,E] or [B,T,E]")
        headlines = headlines.masked_fill(~mask[..., None], 0)
        sentiment = sentiment.masked_fill(~mask[..., None], 0)
        if not all(torch.isfinite(x).all() for x in (headlines, sentiment, market, description)):
            raise ValueError("Active inputs must be finite")
        return headlines, sentiment, market, mask, description, legacy

    def trace(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return live intermediate tensors; does not detach or change eval mode."""
        headlines, sentiment, prices, mask, description, legacy = self._inputs(batch)
        b, t, k, _ = headlines.shape
        market_text, news_text = self.text_projection(description), self.text_projection(headlines)
        market_numbers = self.market_numbers(torch.zeros_like(prices) if self.variant == "news_only" else prices)
        news_numbers = self.news_numbers(sentiment)
        market = self.market_fusion(torch.cat((market_text, market_numbers), dim=-1))
        news = self.news_fusion(torch.cat((news_text, news_numbers), dim=-1))
        news = news.masked_fill(~mask[..., None], 0)
        self_weights = news.new_zeros((b, t, self.attention_heads, k, k))
        if self.variant not in {"no_self_attention", "mean_pool"}:
            h, d = self.attention_heads, self.attention_width // self.attention_heads
            q, key, val = (layer(news).reshape(b, t, k, h, d).transpose(2, 3)
                           for layer in (self.self_query, self.self_key, self.self_value))
            self_scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(d)
            self_weights = self._masked_softmax(self_scores, mask[:, :, None, None, :])
            self_weights = self_weights.masked_fill(~mask[:, :, None, :, None], 0)
            attended = torch.matmul(self.self_dropout(self_weights), val)
            attended = attended.transpose(2, 3).reshape(b, t, k, self.attention_width)
            attended = self.self_output(attended)
            refined = self.normalization(news + attended) if self.attention_style == "residual_norm" else attended
        else:
            refined = news
        refined = refined.masked_fill(~mask[..., None], 0)
        query, news_keys = self.market_query(market), self.news_key(refined)
        scores = (query[:, :, None] * news_keys).sum(dim=-1) / math.sqrt(self.attention_width)
        if self.variant == "mean_pool":
            weights = mask.to(news.dtype) / mask.sum(dim=-1, keepdim=True).clamp_min(1)
        else:
            weights = self._masked_softmax(scores, mask)
        summaries = (weights[..., None] * refined).sum(dim=2)
        predictor_market = torch.zeros_like(market) if self.variant == "news_only" else market
        # Eq. 7 lists complete market history followed by complete news history.
        mlp_input = torch.cat((predictor_market.flatten(1), summaries.flatten(1)), dim=-1)
        logits = self.predictor(mlp_input).squeeze(-1)
        return {
            "market_projected_text": market_text, "news_projected_text": news_text,
            "market_numbers": market_numbers, "news_numbers": news_numbers,
            "market_encoded": market, "news_encoded": news,
            "news_refined": refined, "self_attention_weights": self_weights,
            "market_query": query, "news_keys": news_keys, "query_scores": scores,
            "weights": weights[:, 0] if legacy else weights,
            "day_news_summaries": summaries, "predictor_market": predictor_market,
            "mlp_input": mlp_input, "logits": logits,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        traced = self.trace(batch)
        return traced["logits"], traced["weights"]
