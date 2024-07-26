package com.tabbyml.tabby4eclipse.editor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;

import org.eclipse.core.resources.IFile;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.DocumentEvent;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.text.IDocumentExtension4;
import org.eclipse.jface.text.IDocumentListener;
import org.eclipse.jface.text.ITextSelection;
import org.eclipse.jface.text.ITextViewer;
import org.eclipse.jface.text.TextSelection;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.lsp4e.LSPEclipseUtils;
import org.eclipse.lsp4j.Position;
import org.eclipse.lsp4j.TextDocumentIdentifier;
import org.eclipse.lsp4j.services.LanguageServer;
import org.eclipse.swt.custom.CaretEvent;
import org.eclipse.swt.custom.CaretListener;
import org.eclipse.swt.custom.StyledText;
import org.eclipse.ui.IEditorPart;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.texteditor.ITextEditor;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.lsp.LanguageServerService;
import com.tabbyml.tabby4eclipse.lsp.protocol.ILanguageServer;
import com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionParams;
import com.tabbyml.tabby4eclipse.lsp.protocol.TextDocumentServiceExt;

public class InlineCompletionService {
	public static InlineCompletionService getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final InlineCompletionService INSTANCE = new InlineCompletionService();
	}

	private Logger logger = new Logger("InlineCompletionService");
	private Map<ITextEditor, CaretListener> caretListeners = new HashMap<>();
	private Map<ITextEditor, IDocumentListener> documentListeners = new HashMap<>();
	private InlineCompletionRenderer renderer = new InlineCompletionRenderer();
	private TriggerEvent lastEvent;
	private TriggerEvent pendingEvent;
	private InlineCompletionContext current;

	public void register(ITextEditor textEditor) {
		ITextViewer textViewer = (ITextViewer) textEditor.getAdapter(ITextViewer.class);
		StyledText widget = textViewer.getTextWidget();
		widget.getDisplay().syncExec(() -> {
			CaretListener caretListener = new CaretListener() {
				@Override
				public void caretMoved(CaretEvent event) {
					handleCaretMoved(textEditor, event);
				}
			};
			widget.addCaretListener(caretListener);
			caretListeners.put(textEditor, caretListener);
			logger.info("Created caret listener for TextEditor " + textViewer.toString());
		});

		IDocument document = LSPEclipseUtils.getDocument(textEditor.getEditorInput());
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
		logger.info("Created document listener for TextEditor " + textViewer.toString());
	}

	public void unregister(ITextEditor textEditor) {
		ITextViewer textViewer = (ITextViewer) textEditor.getAdapter(ITextViewer.class);
		StyledText widget = textViewer.getTextWidget();
		widget.getDisplay().syncExec(() -> {
			CaretListener caretListener = caretListeners.get(textEditor);
			if (caretListener != null) {
				widget.removeCaretListener(caretListener);
				caretListeners.remove(textEditor);
				logger.info("Removed caret listener for TextEditor " + textViewer.toString());
			}
		});

		IDocument document = LSPEclipseUtils.getDocument(textEditor.getEditorInput());
		IDocumentListener documentListener = documentListeners.get(textEditor);
		if (documentListener != null) {
			document.removeDocumentListener(documentListener);
			documentListeners.remove(textEditor);
			logger.info("Removed document listener for TextEditor " + textViewer.toString());
		}
	}

	public void provideInlineCompletion(ITextEditor textEditor, int offset, int offsetInWidget) {
		logger.debug("Provide inline completion for TextEditor " + textEditor.toString());
		renderer.hide();
		if (current != null) {
			if (current.job != null && !current.job.isDone()) {
				current.job.cancel(true);
			}
			current = null;
		}
		
		ITextViewer textViewer = (ITextViewer) textEditor.getAdapter(ITextViewer.class);
		
		InlineCompletionContext.Request request = new InlineCompletionContext.Request(textEditor, offset);
		logger.debug("Request request: " + request.offset + "," + offsetInWidget);
		InlineCompletionParams params = request.toInlineCompletionParams();
		if (params == null) {
			logger.debug("Failed to create InlineCompletionParams");
			return;
		}
		Function<LanguageServer, CompletableFuture<com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionList>> jobFn = (
				server) -> {
			TextDocumentServiceExt textDocumentService = ((ILanguageServer) server).getTextDocumentServiceExt();
			return textDocumentService.inlineCompletion(params);
		};
		CompletableFuture<com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionList> job = LanguageServerService
				.getInstance().getServer().execute(jobFn);
		job.thenAccept((completionList) -> {
			if (completionList == null || request != current.request) {
				return;
			}
			try {
				InlineCompletionList list = request.convertInlineCompletionList(completionList);
				current.response = new InlineCompletionContext.Response(list);
				renderer.show(textViewer, offsetInWidget, current.response.getActiveCompletionItem(), request.offset);
			} catch (BadLocationException e) {
				logger.error("Failed to show inline completion.", e);
			}
		});
		InlineCompletionContext context = new InlineCompletionContext(request, job, null);
		current = context;
	}

	public boolean isInlineCompletionVisible() {
		return isInlineCompletionVisible(getActiveEditor());
	}

	public boolean isInlineCompletionVisible(ITextEditor textEditor) {
		ITextViewer textViewer = (ITextViewer) textEditor.getAdapter(ITextViewer.class);
		return current != null && current.request != null && current.request.textEditor == textEditor
				&& current.response != null && textViewer != null && textViewer == renderer.getCurrentTextViewer()
				&& renderer.getCurrentCompletionItem() != null
				&& renderer.getCurrentCompletionItem() == current.response.getActiveCompletionItem();
	}

	public void accept() {
		accept(getActiveEditor());
	}

	public void accept(ITextEditor textEditor) {
		logger.debug("Accept inline completion in TextEditor " + textEditor.toString());
		if (current == null || current.request == null || current.response == null) {
			return;
		}
		ITextViewer textViewer = (ITextViewer) textEditor.getAdapter(ITextViewer.class);
		IDocument document = LSPEclipseUtils.getDocument(textEditor.getEditorInput());
		int offset = current.request.offset;
		InlineCompletionItem item = current.response.getActiveCompletionItem();

		renderer.hide();
		current = null;

		int prefixReplaceLength = offset - item.getReplaceRange().getStart();
		int suffixReplaceLength = item.getReplaceRange().getEnd() - offset;
		String text = item.getInsertText().substring(prefixReplaceLength);
		if (text.isEmpty()) {
			return;
		}

		textViewer.getTextWidget().getDisplay().syncExec(() -> {
			try {
				document.replace(offset, suffixReplaceLength, text);
				ITextSelection selection = new TextSelection(offset + text.length(), 0);
				textEditor.getSelectionProvider().setSelection(selection);
			} catch (BadLocationException e) {
				logger.error("Failed to accept inline completion.", e);
			}
		});
	}

	public void dismiss() {
		logger.debug("Dismiss inline completion in current TextEditor.");
		renderer.hide();
		if (current != null) {
			if (current.job != null && !current.job.isDone()) {
				current.job.cancel(true);
			}
			current = null;
		}
	}

	private void handleCaretMoved(ITextEditor textEditor, CaretEvent event) {
		if (!isActiveEditor(textEditor)) {
			return;
		}
		logger.debug("handleCaretMoved offset:" + event.caretOffset);
		long modificationStamp = getDocumentModificationStamp(textEditor);
		if (pendingEvent != null && pendingEvent.textEditor == textEditor) {
			if (pendingEvent.documentEvent != null && pendingEvent.modificationStamp == modificationStamp) {
				int offsetInWidget = event.caretOffset;
				int offsetDelta = 0;
				if (lastEvent != null && lastEvent.textEditor == pendingEvent.textEditor) {
					offsetDelta = offsetInWidget - lastEvent.caretEvent.caretOffset;
				}
				int offset = getDocumentOffset(textEditor, pendingEvent.documentEvent, offsetDelta);
				provideInlineCompletion(textEditor, offset, offsetInWidget);
				pendingEvent.caretEvent = event;
				lastEvent = pendingEvent;
				pendingEvent = null;
			} else {
				pendingEvent.caretEvent = event;
				pendingEvent.modificationStamp = modificationStamp;
			}
		} else {
			dismiss();
		}
	}

	private void handleDocumentAboutToBeChanged(ITextEditor textEditor, DocumentEvent event) {
		if (!isActiveEditor(textEditor)) {
			return;
		}
		logger.debug("handleDocumentAboutToBeChanged offset:" + event.getOffset());
		pendingEvent = new TriggerEvent();
		pendingEvent.textEditor = textEditor;
	}

	private void handleDocumentChanged(ITextEditor textEditor, DocumentEvent event) {
		if (!isActiveEditor(textEditor)) {
			return;
		}
		logger.debug("handleDocumentChanged offset:" + event.getOffset());
		long modificationStamp = getDocumentModificationStamp(textEditor);
		if (pendingEvent != null && pendingEvent.textEditor == textEditor) {
			if (pendingEvent.caretEvent != null && pendingEvent.modificationStamp == modificationStamp) {
				int offsetInWidget = pendingEvent.caretEvent.caretOffset;
				int offsetDelta = 0;
				if (lastEvent != null && lastEvent.textEditor == pendingEvent.textEditor) {
					offsetDelta = offsetInWidget - lastEvent.caretEvent.caretOffset;
				}
				int offset = getDocumentOffset(textEditor, event, offsetDelta);
				provideInlineCompletion(textEditor, offset, offsetInWidget);
				pendingEvent.documentEvent = event;
				lastEvent = pendingEvent;
				pendingEvent = null;
			} else {
				pendingEvent.documentEvent = event;
				pendingEvent.modificationStamp = modificationStamp;
			}
		}
	}

	private ITextEditor getActiveEditor() {
		IWorkbenchWindow window = PlatformUI.getWorkbench().getActiveWorkbenchWindow();
		if (window != null) {
			IWorkbenchPage page = window.getActivePage();
			if (page != null) {
				IEditorPart activeEditor = page.getActiveEditor();
				if (activeEditor instanceof ITextEditor textEditor) {
					return textEditor;
				}
			}
		}
		return null;
	}

	private boolean isActiveEditor(ITextEditor textEditor) {
		return textEditor == getActiveEditor();
	}
	
	private class TriggerEvent {
		private ITextEditor textEditor;
		private long modificationStamp;
		private DocumentEvent documentEvent;
		private CaretEvent caretEvent;
	}

	private class InlineCompletionContext {
		private static class Request {
			private ITextEditor textEditor;
			private IDocument document;
			private int offset;
			private boolean manually;

			public Request(ITextEditor textEditor, int offset) {
				this.textEditor = textEditor;
				this.document = LSPEclipseUtils.getDocument(textEditor.getEditorInput());
				this.offset = offset;
				this.manually = false;
			}

			private Logger logger = new Logger("toInlineCompletionParams");

			public InlineCompletionParams toInlineCompletionParams() {
				try {
					InlineCompletionParams.InlineCompletionContext context = new InlineCompletionParams.InlineCompletionContext(
							manually ? InlineCompletionParams.InlineCompletionTriggerKind.Invoked
									: InlineCompletionParams.InlineCompletionTriggerKind.Automatic,
							null);
					TextDocumentIdentifier documentIdentifier = LSPEclipseUtils.toTextDocumentIdentifier(document);
					Position position = LSPEclipseUtils.toPosition(offset, document);
					InlineCompletionParams params = new InlineCompletionParams(context, documentIdentifier, position);
					return params;
				} catch (BadLocationException e) {
					logger.debug("Failed to create InlineCompletionParams.");
					return null;
				}
			}

			public InlineCompletionList convertInlineCompletionList(
					com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionList list) throws BadLocationException {
				boolean isIncomplete = list.isIncomplete();
				List<InlineCompletionItem> items = new ArrayList<>();
				for (int i = 0; i < list.getItems().size(); i++) {
					com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionItem item = list.getItems().get(i);
					String insertText = item.getInsertText();
					InlineCompletionItem.Range range = new InlineCompletionItem.Range(
							LSPEclipseUtils.toOffset(item.getRange().getStart(), document),
							LSPEclipseUtils.toOffset(item.getRange().getEnd(), document));
					items.add(new InlineCompletionItem(insertText, range));
				}
				return new InlineCompletionList(isIncomplete, items);
			}
		}

		private static class Response {
			private InlineCompletionList completionList;
			private int itemIndex;

			public Response(InlineCompletionList completionList) {
				this.completionList = completionList;
				this.itemIndex = 0;
			}

			public InlineCompletionItem getActiveCompletionItem() {
				if (itemIndex >= 0 && itemIndex < completionList.getItems().size()) {
					return completionList.getItems().get(itemIndex);
				}
				return null;
			}
		}

		private Request request;
		private CompletableFuture<com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionList> job;
		private Response response;

		public InlineCompletionContext(Request request,
				CompletableFuture<com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionList> job, Response response) {
			this.request = request;
			this.job = job;
			this.response = response;
		}
	}

	private static int getDocumentOffset(ITextEditor textEditor, DocumentEvent event, int delta) {
		int newLength = event.getText().length();
		if (newLength >= 2 && delta > 0) {
			return event.getOffset() + delta;
		} else {
			return event.getOffset() + event.getText().length();
		}
	}
	
	private static long getDocumentModificationStamp(ITextEditor textEditor) {
		IDocument document = LSPEclipseUtils.getDocument(textEditor.getEditorInput());
		if (document instanceof IDocumentExtension4 documentExt) {
			return documentExt.getModificationStamp();
		} else if (document != null) {
			IFile file = LSPEclipseUtils.getFile(document);
			if (file != null) {
				return file.getModificationStamp();
			}
		}
		return IDocumentExtension4.UNKNOWN_MODIFICATION_STAMP;
	}
}
