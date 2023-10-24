package com.tabbyml.codecompletion;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jdt.ui.text.java.ContentAssistInvocationContext;
import org.eclipse.jdt.ui.text.java.IJavaCompletionProposalComputer;
import org.eclipse.jface.text.contentassist.CompletionProposal;
import org.eclipse.jface.text.contentassist.ICompletionProposal;
import org.eclipse.jface.text.contentassist.IContextInformation;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.tabbyml.codecompletion.api.TabbyCompletionService;
import com.tabbyml.codecompletion.api.TabbyV1Choice;
import com.tabbyml.codecompletion.api.TabbyV1CompletionRequest;
import com.tabbyml.codecompletion.api.TabbyV1CompletionResponse;
import com.tabbyml.codecompletion.settings.PreferenceConstants;
import com.tabbyml.codecompletion.settings.PreferenceInitializer;

public class TabbyCompletionProposalComputer implements IJavaCompletionProposalComputer {
	private static Logger LOG = Logger.getLogger(TabbyCompletionProposalComputer.class.getName());
	private static final String apiBase = PreferenceInitializer.getPreferenceStore().getString(PreferenceConstants.P_SERVER_ENDPOINT);
    private static final int displayStringLength = 50;
    private static final TabbyCompletionService tabbyService = new TabbyCompletionService(apiBase);
	
	
    @Override
    public void sessionStarted() {
    }

    @Override
    public String getErrorMessage() {
        return null;
    }

    @Override
    public void sessionEnded() {
    }
    
	@Override
	public List<ICompletionProposal> computeCompletionProposals(ContentAssistInvocationContext context,
			IProgressMonitor monitor) {
		List<ICompletionProposal> proposals = new ArrayList<>();

        try {
        	int currentOffset = context.getInvocationOffset();
            // get document until current offset
            String prefix = context.getDocument().get(0, currentOffset);
            String suffix = context.getDocument().get(currentOffset, context.getDocument().getLength() - currentOffset);

            LOG.info("invoking tabby on context");
            LOG.info(prefix);
            LOG.info(suffix);

            TabbyV1CompletionResponse tabbyProposal = tabbyService.completeWithTabby(prefix, suffix);
            
            if(null != tabbyProposal) {
            	for (TabbyV1Choice choice : tabbyProposal.getChoices()) {
            		String displayString = "TabbyML No." + choice.getIndex() + 1 + ": " + choice.getText().substring(0, choice.getText().length() >= displayStringLength ? displayStringLength : choice.getText().length());
            		String additionalInfoString = "==========Tabby ML==========<br/>"
            										+ "apiBase: " + apiBase + "<br/>"
            										+ "===========================<br/>Request ID: "
            										+ String.valueOf(tabbyProposal.getId())  + "<br/>"
            										+ "===========================<br/>Choice Index: " 
            										+ String.valueOf(choice.getIndex()) + "<br/>"
            										+ "===========================<br/>Proposal:<br/><br/>"
            										+ choice.getText();
            		proposals.add(
            				new CompletionProposal(choice.getText(), // The code snippet to insert
            						context.getInvocationOffset(), // Offset where the completion is applied
            						0, // Length of the text to replace (0 means insert)
            						choice.getText().length(), // Position of the cursor after completion
            						null, displayString, null, 
            						additionalInfoString));
            	}
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return proposals;
	}

	@Override
	public List<IContextInformation> computeContextInformation(ContentAssistInvocationContext context,
			IProgressMonitor monitor) {
		return null;
	}
	
	
}
