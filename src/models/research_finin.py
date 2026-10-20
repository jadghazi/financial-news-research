"""Controlled FININ ablations; original equations remain the concat default.

Temporal pooling is a declared extension: uniform and decay variants pool into
the same news vector, retaining identical complete price histories. Fixed decay
on concatenated day slots is disallowed because a first linear layer can absorb
it. This module does not change frozen historical PaperFININ implementations.

See docs/code_guide.md#headline-attention-and-prediction for the code map, contracts, and execution boundaries.
"""
import math

import torch

from src.models.paper_finin import PaperFININ, _mlp


class ResearchFININ(PaperFININ):
    """Expose independent channel/attention controls and matched temporal pooling.

    With width 16 and five sessions, concat has 160 predictor inputs and pooled
    has 96. ``trace`` exposes the same intermediate tensors used by forward.
    See docs/code_guide.md#headline-attention-and-prediction before adding arms.
    """
    def __init__(self, *, interaction=True, query_weighting=True, news_available=True,
                 sentiment_available=True, prices_available=True,
                 direct_market_available=True,
                 temporal_aggregation='concat', half_life_hours=None, **kwargs):
        if temporal_aggregation not in {'concat', 'pooled'}:
            raise ValueError('Temporal aggregation must be concat or pooled')
        if half_life_hours is not None and (not math.isfinite(half_life_hours) or half_life_hours <= 0):
            raise ValueError('A finite positive half-life is required')
        if half_life_hours is not None and temporal_aggregation != 'pooled':
            raise ValueError('Decay requires pooling; fixed concat weights are absorbable by the MLP')
        if 'variant' in kwargs:
            raise ValueError('Use explicit interaction/query/channel switches for controlled ablations')
        super().__init__(variant='full', **kwargs)
        self.interaction, self.query_weighting = interaction, query_weighting
        self.news_available, self.sentiment_available = news_available, sentiment_available
        self.prices_available = prices_available
        self.direct_market_available = direct_market_available
        self.temporal_aggregation, self.half_life_hours = temporal_aggregation, half_life_hours
        if temporal_aggregation == 'pooled':
            self.predictor = _mlp((self.history_days + 1) * self.width, 1, self.width,
                                  kwargs.get('mlp_layers', 2), kwargs.get('dropout', .1))
            if kwargs.get('initial_probability') is not None:
                probability = kwargs['initial_probability']
                torch.nn.init.zeros_(self.predictor[-1].weight)
                torch.nn.init.constant_(self.predictor[-1].bias, math.log(probability / (1 - probability)))

    def trace(self, batch):
        headlines, sentiment, prices, mask, description, legacy = self._inputs(batch)
        b, t, k, _ = headlines.shape
        if not self.news_available:
            mask = torch.zeros_like(mask)
        headlines = headlines.masked_fill(~mask[..., None], 0)
        sentiment = sentiment.masked_fill(~mask[..., None], 0)
        if not self.sentiment_available:
            sentiment = torch.zeros_like(sentiment)
        market_text, news_text = self.text_projection(description), self.text_projection(headlines)
        market_numbers = self.market_numbers(prices if self.prices_available else torch.zeros_like(prices))
        news_numbers = self.news_numbers(sentiment)
        market = self.market_fusion(torch.cat((market_text, market_numbers), -1))
        news = self.news_fusion(torch.cat((news_text, news_numbers), -1)).masked_fill(~mask[..., None], 0)
        self_weights = news.new_zeros((b, t, self.attention_heads, k, k))
        if self.interaction:
            h, d = self.attention_heads, self.attention_width // self.attention_heads
            q, key, value = (layer(news).reshape(b, t, k, h, d).transpose(2, 3)
                             for layer in (self.self_query, self.self_key, self.self_value))
            scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(d)
            self_weights = self._masked_softmax(scores, mask[:, :, None, None, :])
            self_weights = self_weights.masked_fill(~mask[:, :, None, :, None], 0)
            attended = torch.matmul(self.self_dropout(self_weights), value).transpose(2, 3)
            attended = self.self_output(attended.reshape(b, t, k, self.attention_width))
            refined = self.normalization(news + attended) if self.attention_style == 'residual_norm' else attended
        else:
            refined = news
        refined = refined.masked_fill(~mask[..., None], 0)
        query, keys = self.market_query(market), self.news_key(refined)
        scores = (query[:, :, None] * keys).sum(-1) / math.sqrt(self.attention_width)
        if self.query_weighting:
            weights = self._masked_softmax(scores, mask)
        else:
            weights = mask.to(news.dtype) / mask.sum(-1, keepdim=True).clamp_min(1)
        summaries = (weights[..., None] * refined).sum(2)
        predictor_market = market if self.prices_available and self.direct_market_available else torch.zeros_like(market)
        combined_weights = weights.new_zeros(weights.shape)
        if self.temporal_aggregation == 'concat':
            mlp_input = torch.cat((predictor_market.flatten(1), summaries.flatten(1)), -1)
        else:
            if self.half_life_hours is None:
                combined_weights = weights / mask.any(-1).sum(-1)[:, None, None].clamp_min(1)
            else:
                ages = batch.get('headline_age_hours')
                if ages is None:
                    raise ValueError('Decay requires actual cutoff-relative headline ages')
                if legacy and ages.ndim == 2:
                    ages = ages[:, None]
                if ages.shape != mask.shape or not torch.isfinite(ages[mask]).all() or (ages[mask] < 0).any():
                    raise ValueError('Active ages must be finite, nonnegative, and match the news mask')
                # Log space avoids underflow for old news and preserves exact padding zeros.
                log_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny).log()
                log_weights = log_weights - math.log(2) * ages.masked_fill(~mask, 0) / self.half_life_hours
                combined_weights = self._masked_softmax(log_weights.flatten(1), mask.flatten(1)).reshape_as(weights)
            pooled = (combined_weights[..., None] * refined).sum((1, 2))
            mlp_input = torch.cat((predictor_market.flatten(1), pooled), -1)
        return {'market_projected_text': market_text, 'news_projected_text': news_text,
                'market_numbers': market_numbers, 'news_numbers': news_numbers,
                'market_encoded': market, 'news_encoded': news, 'news_refined': refined,
                'self_attention_weights': self_weights, 'market_query': query, 'news_keys': keys,
                'query_scores': scores, 'weights': weights[:, 0] if legacy else weights,
                'day_news_summaries': summaries, 'temporal_news_weights': combined_weights,
                'predictor_market': predictor_market, 'mlp_input': mlp_input,
                'logits': self.predictor(mlp_input).squeeze(-1)}
