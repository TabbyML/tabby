package com.tabbyml.intellijtabby.lsp.protocol.server

import com.tabbyml.intellijtabby.lsp.protocol.StatusIgnoredIssuesEditParams
import com.tabbyml.intellijtabby.lsp.protocol.StatusInfo
import com.tabbyml.intellijtabby.lsp.protocol.StatusRequestParams
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment
import java.util.concurrent.CompletableFuture

@JsonSegment("tabby")
interface StatusFeature {
  @JsonRequest("status")
  fun getStatus(params: StatusRequestParams): CompletableFuture<StatusInfo>

  @JsonRequest("status/showHelpMessage")
  fun showHelpMessage(params: Any): CompletableFuture<Boolean>

  @JsonRequest("status/ignoredIssues/edit")
  fun editIgnoredIssues(params: StatusIgnoredIssuesEditParams): CompletableFuture<Boolean>
}
