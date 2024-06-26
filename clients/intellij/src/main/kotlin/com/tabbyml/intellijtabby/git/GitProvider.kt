package com.tabbyml.intellijtabby.git

interface GitProvider {
  fun isSupported(): Boolean

  data class Repository(
    val root: String,
    val remotes: List<Remote>?,
  ) {
    data class Remote(
      val name: String,
      val url: String,
    )
  }

  fun getRepository(fileUri: String): Repository? {
    return null
  }

  fun diff(rootUri: String, cached: Boolean = false): List<String>? {
    return null
  }
}