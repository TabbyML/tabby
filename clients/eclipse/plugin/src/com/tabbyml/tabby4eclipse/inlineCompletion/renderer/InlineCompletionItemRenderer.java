package com.tabbyml.tabby4eclipse.inlineCompletion.renderer;

public class InlineCompletionItemRenderer {
	public static IInlineCompletionItemRenderer getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final IInlineCompletionItemRenderer INSTANCE = createInstance();
	}

	private static String TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_TEXTPAINTER = System
			.getenv("TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_TEXTPAINTER");
	private static boolean EXPERIMENTAL_RENDERER_TEXTPAINTER = TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_TEXTPAINTER != null
			&& !TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_TEXTPAINTER.isEmpty();

	private static String TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_CODEMINING = System
			.getenv("TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_CODEMINING");
	private static boolean EXPERIMENTAL_RENDERER_CODEMINING = TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_CODEMINING != null
			&& !TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_CODEMINING.isEmpty();

	public static IInlineCompletionItemRenderer createInstance() {
		if (EXPERIMENTAL_RENDERER_TEXTPAINTER) {
			return new InlineCompletionItemTextPainter();
		}
		return new InlineCompletionItemTextPainter();
	}
}
