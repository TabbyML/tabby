package com.tabbyml.tabby4eclipse.inlineCompletion.renderer;

public class InlineCompletionItemRenderer {
	public static IInlineCompletionItemRenderer getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final IInlineCompletionItemRenderer INSTANCE = createInstance();
	}

	public static IInlineCompletionItemRenderer createInstance() {
		return new InlineCompletionItemTextPainter();
	}
}
