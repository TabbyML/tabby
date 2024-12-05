package com.tabbyml.tabby4eclipse.lsp.protocol;

import org.eclipse.lsp4j.SemanticTokens;
import org.eclipse.lsp4j.SemanticTokensLegend;

public class SemanticTokensRangeResult {
	private SemanticTokensLegend legend;
	private SemanticTokens tokens;

	public SemanticTokensRangeResult() {
	}

	public SemanticTokensLegend getLegend() {
		return legend;
	}

	public void setLegend(SemanticTokensLegend legend) {
		this.legend = legend;
	}

	public SemanticTokens getTokens() {
		return tokens;
	}

	public void setTokens(SemanticTokens tokens) {
		this.tokens = tokens;
	}
}
