// Modified from https://github.com/eclipse/lsp4e/blob/main/org.eclipse.lsp4e/src/org/eclipse/lsp4e/operations/completion/LSContentAssistProcessor.java

/*******************************************************************************
 * Copyright (c) 2016, 2023 Red Hat Inc. and others.
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *
 * Contributors:
 *  Mickael Istria (Red Hat Inc.) - initial implementation
 *  Lucas Bullen (Red Hat Inc.) - Bug 520700 - TextEditors within FormEditors are not supported *   Lucas Bullen (Red Hat Inc.) - Refactored for incomplete completion lists
 *								- Refactored for incomplete completion lists
 *******************************************************************************/

package com.tabbyml.tabby4eclipse.editor;

import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

import org.eclipse.core.runtime.OperationCanceledException;
import org.eclipse.jdt.annotation.NonNull;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.text.ITextViewer;
import org.eclipse.jface.text.contentassist.CompletionProposal;
import org.eclipse.jface.text.contentassist.ContextInformation;
import org.eclipse.jface.text.contentassist.ContextInformationValidator;
import org.eclipse.jface.text.contentassist.ICompletionProposal;
import org.eclipse.jface.text.contentassist.IContentAssistProcessor;
import org.eclipse.jface.text.contentassist.IContextInformation;
import org.eclipse.jface.text.contentassist.IContextInformationValidator;
import org.eclipse.lsp4e.LSPEclipseUtils;
import org.eclipse.lsp4e.LanguageServerPlugin;
import org.eclipse.lsp4e.LanguageServerWrapper;
import org.eclipse.lsp4e.LanguageServers;
import org.eclipse.lsp4e.internal.CancellationSupport;
import org.eclipse.lsp4e.internal.CancellationUtil;
import org.eclipse.lsp4e.operations.completion.LSCompletionProposal;
import org.eclipse.lsp4e.ui.Messages;
import org.eclipse.lsp4e.ui.UI;
import org.eclipse.lsp4j.CompletionItem;
import org.eclipse.lsp4j.CompletionItemDefaults;
import org.eclipse.lsp4j.CompletionList;
import org.eclipse.lsp4j.CompletionOptions;
import org.eclipse.lsp4j.CompletionParams;
import org.eclipse.lsp4j.SignatureHelpOptions;
import org.eclipse.lsp4j.SignatureHelpParams;
import org.eclipse.lsp4j.SignatureInformation;
import org.eclipse.lsp4j.jsonrpc.CancelChecker;
import org.eclipse.lsp4j.jsonrpc.ResponseErrorException;
import org.eclipse.lsp4j.jsonrpc.messages.Either;
import org.eclipse.ui.texteditor.ITextEditor;

import com.tabbyml.tabby4eclipse.lsp.LanguageServerManager;

public class ContentAssistProcessor implements IContentAssistProcessor {

	private static final long TRIGGERS_TIMEOUT = 50;
	private static final long CONTEXT_INFORMATION_TIMEOUT = 1000;
	private IDocument currentDocument;
	private String errorMessage;
	private final boolean errorAsCompletionItem;
	private CompletableFuture<@NonNull List<@NonNull Void>> completionLanguageServersFuture;
	private final Object completionTriggerCharsSemaphore = new Object();
	private char[] completionTriggerChars = new char[0];
	private CompletableFuture<@NonNull List<@NonNull Void>> contextInformationLanguageServersFuture;
	private final Object contextTriggerCharsSemaphore = new Object();
	private char[] contextTriggerChars = new char[0];

	// The cancellation support used to cancel previous LSP requests
	// 'textDocument/completion' when completion is retriggered
	private CancellationSupport cancellationSupport;

	public ContentAssistProcessor() {
		this(true);
	}

	public ContentAssistProcessor(boolean errorAsCompletionItem) {
		this.errorAsCompletionItem = errorAsCompletionItem;
		this.cancellationSupport = new CancellationSupport();
	}

	class CompletionProposalComparator implements Comparator<LSCompletionProposal> {
		@Override
		public int compare(LSCompletionProposal o1, LSCompletionProposal o2) {
			try {
				int docFilterLen1 = o1.getDocumentFilter().length();
				int docFilterLen2 = o2.getDocumentFilter().length();
				if (docFilterLen1 > docFilterLen2) {
					return -1;
				} else if (docFilterLen1 < docFilterLen2) {
					return +1;
				}
			} catch (BadLocationException e) {
				LanguageServerPlugin.logError(e);
			}
			if (o1.getRankCategory() < o2.getRankCategory()) {
				return -1;
			} else if (o1.getRankCategory() > o2.getRankCategory()) {
				return +1;
			}
			if ((o1.getRankCategory() < 5 && o2.getRankCategory() < 5)
					&& (!(o1.getRankScore() == -1 && o2.getRankScore() == -1))) {
				if (o2.getRankScore() == -1 || o1.getRankScore() < o2.getRankScore()) {
					return -1;
				} else if (o1.getRankScore() == -1 || o1.getRankScore() > o2.getRankScore()) {
					return +1;
				}
			}
			String c1 = o1.getSortText();
			String c2 = o2.getSortText();
			if (c1 == null) {
				return -1;
			}
			return c1.compareToIgnoreCase(c2);
		}
	}

	private final Comparator<LSCompletionProposal> proposalComparator = new CompletionProposalComparator();

	@Override
	public ICompletionProposal[] computeCompletionProposals(ITextViewer viewer, int offset) {
		IDocument document = viewer.getDocument();

		URI uri = LSPEclipseUtils.toUri(document);
		if (uri == null) {
			return new LSCompletionProposal[0];
		}

		initiateLanguageServers(document);
		CompletionParams param;

		try {
			param = LSPEclipseUtils.toCompletionParams(uri, offset, document, this.completionTriggerChars);
		} catch (BadLocationException e) {
			LanguageServerPlugin.logError(e);
			this.errorMessage = createErrorMessage(offset, e);
			return createErrorProposal(offset, e);
		}

		List<ICompletionProposal> proposals = Collections.synchronizedList(new ArrayList<>());
		try {
			// Cancel the previous LSP requests 'textDocument/completions' and
			// completionLanguageServersFuture
			this.cancellationSupport.cancel();

			// Initialize a new cancel support to register:
			// - LSP requests 'textDocument/completions'
			// - completionLanguageServersFuture
			CancellationSupport cancellationSupport = new CancellationSupport();
			this.completionLanguageServersFuture = LanguageServers.forDocument(document)
					.withPreferredServer(LanguageServerManager.getLanguageServerDefinition())
					.withFilter(capabilities -> capabilities.getCompletionProvider() != null) //
					.collectAll((w, ls) -> cancellationSupport.execute(ls.getTextDocumentService().completion(param)) //
							.thenAccept(completion -> {
								// Always treat the completion list as complete
								boolean isIncomplete = false;
								proposals.addAll(toProposals(document, offset, completion, w, cancellationSupport,
										isIncomplete));
							}));
			cancellationSupport.execute(completionLanguageServersFuture);
			this.cancellationSupport = cancellationSupport;

			// Wait for the result of all LSP requests 'textDocument/completions', this
			// future will be canceled with the next completion
			this.completionLanguageServersFuture.get();
		} catch (ResponseErrorException | ExecutionException | CancellationException e) {
			if (!CancellationUtil.isRequestCancelledException(e)) { // do not report error if the server has cancelled
																	// the request
				LanguageServerPlugin.logError(e);
			}
			this.errorMessage = createErrorMessage(offset, e);
			return createErrorProposal(offset, e);
		} catch (InterruptedException e) {
			LanguageServerPlugin.logError(e);
			this.errorMessage = createErrorMessage(offset, e);
			Thread.currentThread().interrupt();
			return createErrorProposal(offset, e);
		}

		final var completeProposals = new LSCompletionProposal[proposals.size()];
		int i = 0;
		for (ICompletionProposal proposal : proposals) {
			if (proposal instanceof LSCompletionProposal completeProposal) {
				completeProposals[i] = completeProposal;
				i++;
			} else {
				return proposals.toArray(ICompletionProposal[]::new);
			}
		}
		Arrays.sort(completeProposals, proposalComparator);
		return completeProposals;
	}

	private ICompletionProposal[] createErrorProposal(int offset, Exception ex) {
		if (errorAsCompletionItem) {
			return new ICompletionProposal[] {
					new CompletionProposal("", offset, 0, 0, null, Messages.completionError, null, ex.getMessage()) }; //$NON-NLS-1$
		} else {
			return new ICompletionProposal[0];
		}
	}

	private String createErrorMessage(int offset, Exception ex) {
		return Messages.completionError + " : " + ex.getMessage(); //$NON-NLS-1$
	}

	private void initiateLanguageServers(@NonNull IDocument document) {
		if (currentDocument != document) {
			this.currentDocument = document;
			if (this.completionLanguageServersFuture != null) {
				try {
					this.completionLanguageServersFuture.cancel(true);
				} catch (CancellationException ex) {
					// nothing
				}
			}
			if (this.contextInformationLanguageServersFuture != null) {
				try {
					this.contextInformationLanguageServersFuture.cancel(true);
				} catch (CancellationException ex) {
					// nothing
				}
			}
			this.completionTriggerChars = new char[0];
			this.contextTriggerChars = new char[0];

			this.completionLanguageServersFuture = LanguageServers.forDocument(document)
					.withPreferredServer(LanguageServerManager.getLanguageServerDefinition())
					.withFilter(capabilities -> capabilities.getCompletionProvider() != null).collectAll((w, ls) -> {
						CompletionOptions provider = w.getServerCapabilities().getCompletionProvider();
						synchronized (completionTriggerCharsSemaphore) {
							completionTriggerChars = mergeTriggers(completionTriggerChars,
									provider.getTriggerCharacters());
						}
						return CompletableFuture.completedFuture(null);
					});
			this.contextInformationLanguageServersFuture = LanguageServers.forDocument(document)
					.withPreferredServer(LanguageServerManager.getLanguageServerDefinition())
					.withFilter(capabilities -> capabilities.getSignatureHelpProvider() != null).collectAll((w, ls) -> {
						SignatureHelpOptions provider = w.getServerCapabilities().getSignatureHelpProvider();
						synchronized (contextTriggerCharsSemaphore) {
							contextTriggerChars = mergeTriggers(contextTriggerChars, provider.getTriggerCharacters());
						}
						return CompletableFuture.completedFuture(null);
					});
		}

	}

	private void initiateLanguageServers() {
		ITextEditor textEditor = UI.getActiveTextEditor();
		if (textEditor != null) {
			IDocument document = LSPEclipseUtils.getDocument(textEditor);
			if (document != null) {
				initiateLanguageServers(document);
			}
		}
	}

	private static List<ICompletionProposal> toProposals(IDocument document, int offset,
			Either<List<CompletionItem>, CompletionList> completionList, LanguageServerWrapper languageServerWrapper,
			CancelChecker cancelChecker, boolean isIncomplete) {
		if (completionList == null) {
			return Collections.emptyList();
		}
		// Stop the compute of ICompletionProposal if the completion has been cancelled
		cancelChecker.checkCanceled();
		CompletionItemDefaults defaults = completionList.map(o -> null, CompletionList::getItemDefaults);
		List<CompletionItem> items = completionList.isLeft() ? completionList.getLeft()
				: completionList.getRight().getItems();
		return items.stream() //
				.filter(Objects::nonNull).map(item -> new LSCompletionProposal(document, offset, item, defaults,
						languageServerWrapper, isIncomplete))
				.filter(proposal -> {
					// Stop the compute of ICompletionProposal if the completion has been cancelled
					cancelChecker.checkCanceled();
					return true;
				}).filter(proposal -> proposal.validate(document, offset, null)).map(ICompletionProposal.class::cast)
				.toList();
	}

	@Override
	public IContextInformation[] computeContextInformation(ITextViewer viewer, int offset) {
		IDocument document = viewer.getDocument();
		if (document == null) {
			return new IContextInformation[] { /* TODO? show error in context information */ };
		}
		initiateLanguageServers(document);
		SignatureHelpParams param;
		try {
			param = LSPEclipseUtils.toSignatureHelpParams(offset, document);
		} catch (BadLocationException e) {
			LanguageServerPlugin.logError(e);
			return new IContextInformation[] { /* TODO? show error in context information */ };
		}
		List<IContextInformation> contextInformations = Collections.synchronizedList(new ArrayList<>());
		try {
			this.contextInformationLanguageServersFuture = LanguageServers.forDocument(document)
					.withPreferredServer(LanguageServerManager.getLanguageServerDefinition())
					.withFilter(capabilities -> capabilities.getSignatureHelpProvider() != null)
					.collectAll(ls -> ls.getTextDocumentService().signatureHelp(param).thenAccept(signatureHelp -> {
						if (signatureHelp != null) {
							signatureHelp.getSignatures().stream().map(ContentAssistProcessor::toContextInformation)
									.forEach(contextInformations::add);
						}
					}));
			this.contextInformationLanguageServersFuture.get(CONTEXT_INFORMATION_TIMEOUT, TimeUnit.MILLISECONDS);
		} catch (ResponseErrorException | ExecutionException e) {
			if (!CancellationUtil.isRequestCancelledException(e)) { // do not report error if the server has cancelled
																	// the request
				LanguageServerPlugin.logError(e);
			}
			return new IContextInformation[] { /* TODO? show error in context information */ };
		} catch (InterruptedException e) {
			LanguageServerPlugin.logError(e);
			Thread.currentThread().interrupt();
			return new IContextInformation[] { /* TODO? show error in context information */ };
		} catch (TimeoutException e) {
			LanguageServerPlugin.logWarning("Could not compute  context information due to timeout after " //$NON-NLS-1$
					+ CONTEXT_INFORMATION_TIMEOUT + " milliseconds", e); //$NON-NLS-1$
			return new IContextInformation[] { /* TODO? show error in context information */ };
		}
		return contextInformations.toArray(IContextInformation[]::new);
	}

	private static IContextInformation toContextInformation(SignatureInformation information) {
		final var signature = new StringBuilder(information.getLabel());
		String docString = LSPEclipseUtils.getDocString(information.getDocumentation());
		if (docString != null && !docString.isEmpty()) {
			signature.append('\n').append(docString);
		}
		return new ContextInformation(information.getLabel(), signature.toString());
	}

	private void getFuture(CompletableFuture<@NonNull List<@NonNull Void>> future) {
		if (future == null) {
			return;
		}

		try {
			future.get(TRIGGERS_TIMEOUT, TimeUnit.MILLISECONDS);
		} catch (InterruptedException e) {
			LanguageServerPlugin.logError(e);
			Thread.currentThread().interrupt();
		} catch (TimeoutException e) {
			LanguageServerPlugin.logWarning(
					"Could not get trigger characters due to timeout after " + TRIGGERS_TIMEOUT + " milliseconds", e); //$NON-NLS-1$//$NON-NLS-2$
		} catch (OperationCanceledException | ResponseErrorException | ExecutionException | CancellationException e) {
			if (!CancellationUtil.isRequestCancelledException(e)) { // do not report error if the server has cancelled
																	// the request
				LanguageServerPlugin.logError(e);
			}
		}
	}

	private static char[] mergeTriggers(char[] initialArray, Collection<String> additionalTriggers) {
		if (initialArray == null) {
			initialArray = new char[0];
		}
		if (additionalTriggers == null) {
			additionalTriggers = Collections.emptySet();
		}
		final var triggers = new HashSet<Character>(initialArray.length);
		for (char c : initialArray) {
			triggers.add(c);
		}
		additionalTriggers.stream().filter(s -> s != null && s.length() > 0).map(triggerChar -> triggerChar.charAt(0))
				.forEach(triggers::add);
		char[] res = new char[triggers.size()];
		int i = 0;
		for (Character c : triggers) {
			res[i] = c;
			i++;
		}
		return res;
	}

	@Override
	public char[] getCompletionProposalAutoActivationCharacters() {
		initiateLanguageServers();
		getFuture(completionLanguageServersFuture);
		return completionTriggerChars;
	}

	@Override
	public char[] getContextInformationAutoActivationCharacters() {
		initiateLanguageServers();
		getFuture(contextInformationLanguageServersFuture);
		return contextTriggerChars;
	}

	@Override
	public String getErrorMessage() {
		return this.errorMessage;
	}

	@Override
	public IContextInformationValidator getContextInformationValidator() {
		return new ContextInformationValidator(this);
	}
}