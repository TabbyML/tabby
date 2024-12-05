package com.tabbyml.tabby4eclipse.inlineCompletion;

public interface IInlineCompletionService {
	/**
	 * Returns whether the completion item ghost text is currently visible in the
	 * active text editor.
	 */
	public boolean isCompletionItemVisible();

	/**
	 * Returns whether the completion request is running.
	 */
	public boolean isLoading();

	/**
	 * Validate the current completion context is still valid for the current caret
	 * position in the active text editor.
	 */
	public boolean isValid();

	/**
	 * Trigger an inline completion request at the current caret position of the
	 * active text editor.
	 *
	 * @param isManualTrigger Whether to trigger manually or automatically.
	 *                        Automatic trigger will only generate one choice, and
	 *                        manual trigger will generate multiple choices.
	 */
	public void trigger(boolean isManualTrigger);

	/**
	 * Cycle to the next choice if it exists. If the current inline completion is
	 * auto-triggered, a manual trigger will be called to attempt to generate
	 * multiple choices.
	 */
	public void next();

	/**
	 * Cycle to the previous choice if it exists. If the current inline completion
	 * is auto-triggered, a manual trigger will be called to attempt to generate
	 * multiple choices.
	 */
	public void previous();

	enum AcceptType {
		FULL_COMPLETION, NEXT_LINE, NEXT_WORD,
	}

	/**
	 * Accept the current completion item ghost text.
	 * 
	 * @param type the type of completion to accept.
	 */
	public void accept(AcceptType type);

	/**
	 * Dismiss the current completion item ghost text.
	 */
	public void dismiss();
}
