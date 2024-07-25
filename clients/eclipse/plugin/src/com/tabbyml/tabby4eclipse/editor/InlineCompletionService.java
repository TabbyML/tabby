package com.tabbyml.tabby4eclipse.editor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
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
import org.eclipse.lsp4e.LSPEclipseUtils;
import org.eclipse.lsp4e.LanguageServerWrapper;
import org.eclipse.lsp4j.Position;
import org.eclipse.lsp4j.services.LanguageServer;
import org.eclipse.lsp4j.TextDocumentIdentifier;
import org.eclipse.swt.custom.CaretEvent;
import org.eclipse.swt.custom.CaretListener;
import org.eclipse.swt.custom.StyledText;
import org.eclipse.ui.IEditorPart;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.texteditor.ITextEditor;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.editor.InlineCompletionItem.Range;
import com.tabbyml.tabby4eclipse.lsp.LanguageServerService;
import com.tabbyml.tabby4eclipse.lsp.protocol.ILanguageServer;
import com.tabbyml.tabby4eclipse.lsp.protocol.TextDocumentServiceExt;
import com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionParams;

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

	public void provideInlineCompletion(ITextEditor textEditor, int offset) {
		logger.debug("Provide inline completion for TextEditor " + textEditor.toString());
		ITextViewer textViewer = (ITextViewer) textEditor.getAdapter(ITextViewer.class);
		if (current != null) {
			current.getJob().cancel(true);
			current = null;
		}
		InlineCompletionContext.Request request = new InlineCompletionContext.Request(textEditor, offset);
		InlineCompletionParams params = request.toInlineCompletionParams();
		if (params == null) {
			logger.debug("Failed to create InlineCompletionParams");
			return;
		}
		logger.debug("Request params: " + params.toString());
		Function<LanguageServer, CompletableFuture<com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionList>> jobFn = (server) -> {
			TextDocumentServiceExt textDocumentService = ((ILanguageServer) server).getTextDocumentServiceExt();
			return textDocumentService.inlineCompletion(params);
		};
		CompletableFuture<com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionList> job = LanguageServerService.getInstance().getServer().execute(jobFn);
		job.thenAccept((completionList) -> {
			if (completionList == null || request != current.request) {
				return;
			}
			try {
				InlineCompletionList list = request.convertInlineCompletionList(completionList);
				current.response = new InlineCompletionContext.Response(list);
				renderer.show(textViewer, offset, current.response.getActiveCompletionItem());
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
		return current != null
				&& current.request != null
				&& current.request.textEditor == textEditor
				&& current.response != null
				&& textViewer != null
				&& textViewer == renderer.getCurrentTextViewer()
				&& renderer.getCurrentCompletionItem() != null
				&& renderer.getCurrentCompletionItem() == current.response.getActiveCompletionItem();
	}
	

	public void accept() {
		accept(getActiveEditor());
	}
	
	public void accept(ITextEditor textEditor) {
		if (current == null || current.request == null || current.response == null) {
			return;
		}
		ITextViewer textViewer = (ITextViewer) textEditor.getAdapter(ITextViewer.class);
		IDocument document = LSPEclipseUtils.getDocument(textEditor.getEditorInput());
		int offset = current.request.offset;
		InlineCompletionItem item = current.response.getActiveCompletionItem();
		
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
		
		renderer.hide();
	}

	public void dismiss() {
		renderer.hide();
	}

	private void handleCaretMoved(ITextEditor textEditor, CaretEvent event) {
		if (!isActiveEditor(textEditor)) {
			return;
		}
		if (current != null && current.isMatch(textEditor, event.caretOffset)) {
			return;
		} else {
			dismiss();
		}
	}

	private void handleDocumentAboutToBeChanged(ITextEditor textEditor, DocumentEvent event) {
	}
	
	private void handleDocumentChanged(ITextEditor textEditor, DocumentEvent event) {
		if (!isActiveEditor(textEditor)) {
			return;
		}
		if (event.getLength() == 0 || event.getText().isEmpty()) {
			// A input or delete action
			dismiss();
			provideInlineCompletion(textEditor, event.getOffset() + event.getText().length());
		}
	}
	
	private ITextEditor getActiveEditor() {
		IWorkbenchWindow window = PlatformUI.getWorkbench().getActiveWorkbenchWindow();
		if (window != null) {
			IWorkbenchPage page = window.getActivePage();
			if (page != null) {
				IEditorPart activeEditor = page.getActiveEditor();
				if (activeEditor instanceof ITextEditor) {
					return (ITextEditor)activeEditor;
				}
			}
		}
		return null;
	}
	
	private boolean isActiveEditor(ITextEditor textEditor) {
		return textEditor == getActiveEditor();
	}

	private class InlineCompletionContext {
		private static class Request {
			private ITextEditor textEditor;
			private IDocument document;
			private long modificationStamp;
			private int offset;
			private boolean manually;

			public Request(ITextEditor textEditor) {
				this(textEditor, textEditor.getAdapter(ITextViewer.class).getTextWidget().getCaretOffset());
			}
			
			public Request(ITextEditor textEditor, int offset) {
				this.textEditor = textEditor;
				this.document = LSPEclipseUtils.getDocument(textEditor.getEditorInput());
				this.modificationStamp = getDocumentModificationStamp(document);
				this.offset = offset;
				this.manually = false;
			}

			public ITextEditor getTextEditor() {
				return textEditor;
			}

			public IDocument getDocument() {
				return document;
			}

			public long getModificationStamp() {
				return modificationStamp;
			}

			public int getOffset() {
				return offset;
			}

			public boolean isManually() {
				return manually;
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

			public InlineCompletionList getCompletionList() {
				return completionList;
			}

			public int getItemIndex() {
				return itemIndex;
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

		public InlineCompletionContext(Request request, CompletableFuture<com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionList> job, Response response) {
			this.request = request;
			this.job = job;
			this.response = response;
		}

		public Request getRequest() {
			return request;
		}

		public CompletableFuture<com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionList> getJob() {
			return job;
		}

		public Response getResponse() {
			return response;
		}

		public boolean isMatch(ITextEditor textEditor, int offset) {
			IDocument document = LSPEclipseUtils.getDocument(textEditor.getEditorInput());
			long modificationStamp = getDocumentModificationStamp(document);
			return request.textEditor == textEditor && request.document == document
					&& request.modificationStamp == modificationStamp && request.offset == offset;
		}

		private static long getDocumentModificationStamp(IDocument document) {
			if (document instanceof IDocumentExtension4 ext) {
				return ext.getModificationStamp();
			} else if (document != null) {
				IFile file = LSPEclipseUtils.getFile(document);
				if (file != null) {
					return file.getModificationStamp();
				}
			}
			return IDocumentExtension4.UNKNOWN_MODIFICATION_STAMP;
		}
	}
}
