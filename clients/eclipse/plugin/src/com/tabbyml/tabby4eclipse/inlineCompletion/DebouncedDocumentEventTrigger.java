package com.tabbyml.tabby4eclipse.inlineCompletion;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import org.eclipse.jface.text.DocumentEvent;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.text.IDocumentListener;
import org.eclipse.swt.custom.CaretEvent;
import org.eclipse.swt.custom.CaretListener;
import org.eclipse.swt.custom.StyledText;
import org.eclipse.ui.texteditor.ITextEditor;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.editor.EditorUtils;

/**
 * This DocumentBasedTrigger listens to document and caret events to determine
 * when to trigger inline completion. When a document changed event is received,
 * it waits for a debouncing interval, then triggers inline completion. When a
 * caret event is received, it waits for a debouncing interval, then checks if
 * the current completion context is valid, and dismisses completion if not
 * valid.
 */
public class DebouncedDocumentEventTrigger implements IInlineCompletionTrigger {
	private final static int DEBOUNCE_INTERVAL = 20; // ms

	private Logger logger = new Logger("InlineCompletionTrigger.DebouncedDocumentEventTrigger");

	private IInlineCompletionService inlineCompletionService = InlineCompletionService.getInstance();

	private Map<ITextEditor, CaretListener> caretListeners = new HashMap<>();
	private Map<ITextEditor, IDocumentListener> documentListeners = new HashMap<>();

	@Override
	public void register(ITextEditor textEditor) {
		StyledText widget = EditorUtils.getStyledTextWidget(textEditor);
		widget.getDisplay().syncExec(() -> {
			CaretListener caretListener = new CaretListener() {
				@Override
				public void caretMoved(CaretEvent event) {
					handleCaretMoved(textEditor, event);
				}
			};
			widget.addCaretListener(caretListener);
			caretListeners.put(textEditor, caretListener);
			logger.debug("Created caret listener for TextEditor " + textEditor.toString());
		});

		IDocument document = EditorUtils.getDocument(textEditor);
		IDocumentListener documentListener = new IDocumentListener() {
			@Override
			public void documentAboutToBeChanged(DocumentEvent event) {
			}

			@Override
			public void documentChanged(DocumentEvent event) {
				handleDocumentChanged(textEditor, event);
			}
		};
		document.addDocumentListener(documentListener);
		documentListeners.put(textEditor, documentListener);
		logger.debug("Created document listener for TextEditor " + textEditor.toString());
	}

	@Override
	public void unregister(ITextEditor textEditor) {
		StyledText widget = EditorUtils.getStyledTextWidget(textEditor);
		widget.getDisplay().syncExec(() -> {
			CaretListener caretListener = caretListeners.get(textEditor);
			if (caretListener != null) {
				widget.removeCaretListener(caretListener);
				caretListeners.remove(textEditor);
				logger.debug("Removed caret listener for TextEditor " + textEditor.toString());
			}
		});

		IDocument document = EditorUtils.getDocument(textEditor);
		IDocumentListener documentListener = documentListeners.get(textEditor);
		if (documentListener != null) {
			document.removeDocumentListener(documentListener);
			documentListeners.remove(textEditor);
			logger.debug("Removed document listener for TextEditor " + textEditor.toString());
		}
	}

	private DebouncedRunnable documentChangedRunnable = new DebouncedRunnable(() -> {
		try {
			EditorUtils.syncExec(() -> {
				logger.debug("Trigger inline completion after debouncing.");
				inlineCompletionService.trigger(false);
			});
		} catch (Exception e) {
			logger.error("Failed to handle documentChangedRunnable after debouncing.", e);
		}
	}, DEBOUNCE_INTERVAL);

	private DebouncedRunnable caretMovedRunnable = new DebouncedRunnable(() -> {
		try {
			EditorUtils.syncExec(() -> {
				if (!inlineCompletionService.isValid()) {
					logger.debug("Dismiss invalid inline completion after debouncing.");
					inlineCompletionService.dismiss();
				} else {
					logger.debug("Keep still valid inline completion after debouncing.");
				}
			});
		} catch (Exception e) {
			logger.error("Failed to handle caretMovedRunnable after debouncing.", e);
		}
	}, DEBOUNCE_INTERVAL);

	private void handleCaretMoved(ITextEditor textEditor, CaretEvent event) {
		if (!EditorUtils.isActiveEditor(textEditor)) {
			return;
		}
		logger.debug("handleCaretMoved: " + event.toString() + " offset: " + event.caretOffset);
		caretMovedRunnable.call();
	}

	private void handleDocumentChanged(ITextEditor textEditor, DocumentEvent event) {
		if (!EditorUtils.isActiveEditor(textEditor)) {
			return;
		}
		logger.debug("handleDocumentChanged: " + event.toString());
		documentChangedRunnable.call();
	}

	private class DebouncedRunnable {
		private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
		private ScheduledFuture<?> future;
		private final long delay;
		private final Runnable task;

		public DebouncedRunnable(Runnable task, long delay) {
			this.task = task;
			this.delay = delay;
		}

		public synchronized void call() {
			if (future != null && !future.isDone()) {
				future.cancel(true);
			}
			future = scheduler.schedule(task, delay, TimeUnit.MILLISECONDS);
		}

		// FIXME: scheduler shutdown not called
		public void shutdown() {
			scheduler.shutdown();
		}
	}
}
