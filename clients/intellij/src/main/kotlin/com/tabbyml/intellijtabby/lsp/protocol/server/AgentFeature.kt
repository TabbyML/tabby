package com.tabbyml.intellijtabby.lsp.protocol.server

import com.tabbyml.intellijtabby.lsp.protocol.IssueDetailParams
import com.tabbyml.intellijtabby.lsp.protocol.IssueDetailResult
import com.tabbyml.intellijtabby.lsp.protocol.IssueList
import com.tabbyml.intellijtabby.lsp.protocol.ServerInfo
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment
import java.util.concurrent.CompletableFuture

@JsonSegment("tabby/agent")
interface AgentFeature {
  @JsonRequest
  fun serverInfo(): CompletableFuture<ServerInfo>

  @JsonRequest
  fun status(): CompletableFuture<String>

  @JsonRequest
  fun issues(): CompletableFuture<IssueList>

  @JsonRequest(value = "issue/detail")
  fun getIssueDetail(params: IssueDetailParams): CompletableFuture<IssueDetailResult?>
}
