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

	private static String TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_TEXTPAINTER2 = System
			.getenv("TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_TEXTPAINTER2");
	private static boolean EXPERIMENTAL_RENDERER_TEXTPAINTER2 = TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_TEXTPAINTER2 != null
			&& !TABBY4ECLIPSE_EXPERIMENTAL_RENDERER_TEXTPAINTER2.isEmpty();

	public static IInlineCompletionItemRenderer createInstance() {
		if (EXPERIMENTAL_RENDERER_TEXTPAINTER) {
			return new InlineCompletionItemTextPainter();
		}
		if (EXPERIMENTAL_RENDERER_TEXTPAINTER2) {
			return new InlineCompletionItemTextPainter2();
		}
		return new InlineCompletionItemTextPainter2();
	}
}
