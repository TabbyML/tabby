package com.tabbyml.intellijtabby.git

class DummyGitProvider : GitProvider {
  override fun isSupported(): Boolean {
    return false
  }
}