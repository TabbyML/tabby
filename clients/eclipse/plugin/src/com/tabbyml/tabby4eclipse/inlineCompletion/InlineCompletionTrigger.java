package com.tabbyml.tabby4eclipse.inlineCompletion;

public class InlineCompletionTrigger {
	public static IInlineCompletionTrigger getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final IInlineCompletionTrigger INSTANCE = createInstance();
	}

	private static String TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_BASICINPUT = System
			.getenv("TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_BASICINPUT");
	private static boolean EXPERIMENTAL_TRIGGER_BASICINPUT = TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_BASICINPUT != null
			&& !TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_BASICINPUT.isEmpty();

	private static String TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_DEBOUNCEDDOCUMENT = System
			.getenv("TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_DEBOUNCEDDOCUMENT");
	private static boolean EXPERIMENTAL_TRIGGER_DEBOUNCEDDOCUMENT = TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_DEBOUNCEDDOCUMENT != null
			&& !TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_DEBOUNCEDDOCUMENT.isEmpty();

	private static String TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_PAIREDDOCUMENT = System
			.getenv("TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_PAIREDDOCUMENT");
	private static boolean EXPERIMENTAL_TRIGGER_PAIREDDOCUMENT = TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_PAIREDDOCUMENT != null
			&& !TABBY4ECLIPSE_EXPERIMENTAL_TRIGGER_PAIREDDOCUMENT.isEmpty();

	public static IInlineCompletionTrigger createInstance() {
		if (EXPERIMENTAL_TRIGGER_BASICINPUT) {
			return new BasicInputEventTrigger();
		}
		if (EXPERIMENTAL_TRIGGER_DEBOUNCEDDOCUMENT) {
			return new DebouncedDocumentEventTrigger();
		}
		if (EXPERIMENTAL_TRIGGER_PAIREDDOCUMENT) {
			return new PairedDocumentEventTrigger();
		}
		return new BasicInputEventTrigger();
	}
}
