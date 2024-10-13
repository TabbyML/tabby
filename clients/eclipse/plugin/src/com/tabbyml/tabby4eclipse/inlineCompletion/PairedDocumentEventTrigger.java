package com.tabbyml.tabby4eclipse.inlineCompletion;

import java.util.HashMap;
import java.util.Map;

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
 * when to trigger inline completion. When a pair of document and caret events
 * are received, which means user is typing, it triggers inline completion. When
 * a single event is received, which means user is moving cursor, it dismisses
 * the current inline completion.
 */
public class PairedDocumentEventTrigger implements IInlineCompletionTrigger {
	private Logger logger = new Logger("InlineCompletionTrigger.PairedDocumentEventTrigger");

	private IInlineCompletionService inlineCompletionService = InlineCompletionService.getInstance();

	private Map<ITextEditor, CaretListener> caretListeners = new HashMap<>();
	private Map<ITextEditor, IDocumentListener> documentListeners = new HashMap<>();
	private TriggerEvent pendingEvent;

	private class TriggerEvent {
		private ITextEditor textEditor;
		private long modificationStamp;
		private DocumentEvent documentEvent;
		private CaretEvent caretEvent;
	}

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
				handleDocumentAboutToBeChanged(textEditor, event);
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

	private void handleCaretMoved(ITextEditor textEditor, CaretEvent event) {
		if (!EditorUtils.isActiveEditor(textEditor)) {
			return;
		}
		logger.debug("handleCaretMoved: " + event.toString() + " offset: " + event.caretOffset);
		long modificationStamp = EditorUtils.getDocumentModificationStamp(textEditor);
		if (pendingEvent != null && pendingEvent.textEditor == textEditor) {
			if (pendingEvent.documentEvent != null && pendingEvent.modificationStamp == modificationStamp) {
				logger.debug("Received caretEvent with paired documentEvent, trigger inline completion.");
				inlineCompletionService.trigger(false);
				pendingEvent = null;
			} else {
				logger.debug("Received caretEvent, waiting for paired documentEvent.");
				pendingEvent.caretEvent = event;
				pendingEvent.modificationStamp = modificationStamp;
			}
		} else {
			logger.debug("Received caretEvent without document changes, dismiss inline completion.");
			inlineCompletionService.dismiss();
		}
	}

	private void handleDocumentAboutToBeChanged(ITextEditor textEditor, DocumentEvent event) {
		if (!EditorUtils.isActiveEditor(textEditor)) {
			return;
		}
		logger.debug("handleDocumentAboutToBeChanged: " + event.toString());
		pendingEvent = new TriggerEvent();
		pendingEvent.textEditor = textEditor;
	}

	private void handleDocumentChanged(ITextEditor textEditor, DocumentEvent event) {
		if (!EditorUtils.isActiveEditor(textEditor)) {
			return;
		}
		logger.debug("handleDocumentChanged: " + event.toString());
		long modificationStamp = EditorUtils.getDocumentModificationStamp(textEditor);
		if (pendingEvent != null && pendingEvent.textEditor == textEditor) {
			if (pendingEvent.caretEvent != null && pendingEvent.modificationStamp == modificationStamp) {
				logger.debug("Received documentEvent with paired caretEvent, trigger inline completion.");
				inlineCompletionService.trigger(false);
				pendingEvent = null;
			} else {
				logger.debug("Received documentEvent, waiting for paired caretEvent.");
				pendingEvent.documentEvent = event;
				pendingEvent.modificationStamp = modificationStamp;
			}
		}
	}

}
