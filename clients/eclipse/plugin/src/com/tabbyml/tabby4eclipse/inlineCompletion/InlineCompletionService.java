package com.tabbyml.tabby4eclipse.inlineCompletion;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.text.ITextSelection;
import org.eclipse.jface.text.ITextViewer;
import org.eclipse.jface.text.TextSelection;
import org.eclipse.lsp4e.LSPEclipseUtils;
import org.eclipse.lsp4j.Position;
import org.eclipse.lsp4j.TextDocumentIdentifier;
import org.eclipse.ui.texteditor.ITextEditor;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.editor.EditorUtils;
import com.tabbyml.tabby4eclipse.lsp.LanguageServerService;
import com.tabbyml.tabby4eclipse.lsp.protocol.CompletionEventId;
import com.tabbyml.tabby4eclipse.lsp.protocol.EventParams;
import com.tabbyml.tabby4eclipse.lsp.protocol.ILanguageServer;
import com.tabbyml.tabby4eclipse.lsp.protocol.ITelemetryService;
import com.tabbyml.tabby4eclipse.lsp.protocol.ITextDocumentServiceExt;
import com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionParams;
import com.tabbyml.tabby4eclipse.preferences.PreferencesService;

public class InlineCompletionService implements IInlineCompletionService {
	public static IInlineCompletionService getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final IInlineCompletionService INSTANCE = new InlineCompletionService();
	}

	private Logger logger = new Logger("InlineCompletionService");
	private InlineCompletionRenderer renderer = new InlineCompletionRenderer();
	private InlineCompletionContext current;

	@Override
	public boolean isCompletionItemVisible() {
		ITextEditor textEditor = EditorUtils.getActiveTextEditor();
		ITextViewer textViewer = EditorUtils.getTextViewer(textEditor);
		return current != null && current.request != null && current.request.textEditor == textEditor
				&& current.response != null && textViewer != null && textViewer == renderer.getCurrentTextViewer()
				&& renderer.getCurrentCompletionItem() != null
				&& renderer.getCurrentCompletionItem() == current.response.getActiveCompletionItem();
	}

	@Override
	public boolean isLoading() {
		return current != null && current.job != null && !current.job.isDone();
	}

	@Override
	public boolean isValid() {
		ITextEditor textEditor = EditorUtils.getActiveTextEditor();
		int offset = EditorUtils.getCurrentOffsetInDocument(textEditor);
		long modificationStamp = EditorUtils.getDocumentModificationStamp(textEditor);
		return current != null && current.request != null && current.request.textEditor == textEditor
				&& current.request.offset == offset && current.request.modificationStamp == modificationStamp;
	}

	@Override
	public void trigger(boolean isManualTrigger) {
		boolean autoTriggerEnabled = PreferencesService.getInstance().getInlineCompletionTriggerAuto();
		if (!autoTriggerEnabled && !isManualTrigger) {
			return;
		}
		ITextEditor textEditor = EditorUtils.getActiveTextEditor();
		int offset = EditorUtils.getCurrentOffsetInDocument(textEditor);
		long modificationStamp = EditorUtils.getDocumentModificationStamp(textEditor);
		logger.info("Provide inline completion for TextEditor " + textEditor.getTitle() + " at offset " + offset
				+ " with modification stamp " + modificationStamp);
		renderer.hide();
		if (current != null) {
			if (current.job != null && !current.job.isDone()) {
				logger.info("Cancel the current job due to new request.");
				current.job.cancel(true);
			}
			current = null;
		}

		ITextViewer textViewer = EditorUtils.getTextViewer(textEditor);
		InlineCompletionContext.Request request = new InlineCompletionContext.Request(textEditor, offset,
				modificationStamp, isManualTrigger);
		InlineCompletionParams params = request.toInlineCompletionParams();
		if (params == null) {
			return;
		}
		CompletableFuture<com.tabbyml.tabby4eclipse.lsp.protocol.InlineCompletionList> job = LanguageServerService
				.getInstance().getServer().execute((server) -> {
					ITextDocumentServiceExt textDocumentService = ((ILanguageServer) server)
							.getTextDocumentServiceExt();
					return textDocumentService.inlineCompletion(params);
				});
		job.thenAccept((completionList) -> {
			if (completionList == null || request != current.request) {
				return;
			}
			try {
				InlineCompletionList list = request.convertInlineCompletionList(completionList);
				current.response = new InlineCompletionContext.Response(list);
				renderer.show(textViewer, current.response.getActiveCompletionItem());
				EventParams eventParams = buildTelemetryEventParams(EventParams.Type.VIEW);
				postTelemetryEvent(eventParams);
			} catch (BadLocationException e) {
				logger.error("Failed to show inline completion.", e);
			}
		});
		InlineCompletionContext context = new InlineCompletionContext(request, job, null);
		current = context;
	}

	@Override
	public void accept() {
		ITextEditor textEditor = EditorUtils.getActiveTextEditor();
		logger.info("Accept inline completion in TextEditor " + textEditor.toString());
		if (current == null || current.request == null || current.response == null) {
			return;
		}
		int offset = current.request.offset;
		InlineCompletionItem item = current.response.getActiveCompletionItem();
		EventParams eventParams = buildTelemetryEventParams(EventParams.Type.SELECT);

		renderer.hide();
		current = null;

		int prefixReplaceLength = item.getReplaceRange().getPrefixLength();
		int suffixReplaceLength = item.getReplaceRange().getSuffixLength();
		String text = item.getInsertText().substring(prefixReplaceLength);
		if (text.isEmpty()) {
			return;
		}

		IDocument document = EditorUtils.getDocument(textEditor);
		EditorUtils.syncExec(textEditor, () -> {
			try {
				document.replace(offset, suffixReplaceLength, text);
				ITextSelection selection = new TextSelection(offset + text.length(), 0);
				textEditor.getSelectionProvider().setSelection(selection);
				postTelemetryEvent(eventParams);
			} catch (BadLocationException e) {
				logger.error("Failed to accept inline completion.", e);
			}
		});
	}

	@Override
	public void dismiss() {
		if (renderer.getCurrentCompletionItem() != null) {
			logger.info("Dismiss inline completion.");
			EventParams eventParams = buildTelemetryEventParams(EventParams.Type.DISMISS);
			renderer.hide();
			postTelemetryEvent(eventParams);
		}
		if (current != null) {
			if (current.job != null && !current.job.isDone()) {
				logger.info("Cancel the current job due to dismissed.");
				current.job.cancel(true);
			}
			current = null;
		}
	}

	private EventParams buildTelemetryEventParams(String type) {
		return buildTelemetryEventParams(type, null);
	}

	private EventParams buildTelemetryEventParams(String type, String selectKind) {
		InlineCompletionItem item = this.renderer.getCurrentCompletionItem();
		if (item != null && item == current.response.getActiveCompletionItem()) {
			EventParams params = new EventParams();
			params.setType(type);
			params.setSelectKind(selectKind);
			params.setCompletionEventId(item.getEventId());
			params.setViewId(this.renderer.getCurrentViewId());
			params.setElapsed(this.renderer.getCurrentDisplayedTime());
			return params;
		}
		return null;
	}

	private void postTelemetryEvent(EventParams params) {
		if (params != null) {
			LanguageServerService.getInstance().getServer().execute((server) -> {
				ITelemetryService telemetryService = ((ILanguageServer) server).getTelemetryService();
				telemetryService.event(params);
				return null;
			});
		}
	}

	private class InlineCompletionContext {
		private static class Request {
			private Logger logger = new Logger("InlineCompletionContext.Request");

			private ITextEditor textEditor;
			private IDocument document;
			private int offset;
			private long modificationStamp;
			private boolean manually;

			public Request(ITextEditor textEditor, int offset, long modificationStamp, boolean manually) {
				this.textEditor = textEditor;
				this.document = EditorUtils.getDocument(textEditor);
				this.offset = offset;
				this.modificationStamp = modificationStamp;
				this.manually = manually;
			}

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
					logger.error("Failed to create InlineCompletionParams.", e);
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
					int prefixReplaceLength = offset - LSPEclipseUtils.toOffset(item.getRange().getStart(), document);
					int suffixReplaceLength = LSPEclipseUtils.toOffset(item.getRange().getEnd(), document) - offset;
					InlineCompletionItem.ReplaceRange replaceRange = new InlineCompletionItem.ReplaceRange(
							prefixReplaceLength, suffixReplaceLength);
					CompletionEventId eventId = null;
					if (item.getData() != null) {
						eventId = item.getData().getEventId();
					}
					items.add(new InlineCompletionItem(insertText, replaceRange, eventId));
					logger.debug("Converted InlineCompletionItem " + i + ": " + insertText + "\n replace range: "
							+ replaceRange.getPrefixLength() + ", " + replaceRange.getSuffixLength());
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
}
