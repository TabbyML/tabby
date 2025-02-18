package com.tabbyml.intellijtabby.chat

import com.google.gson.Gson

abstract class Filepath(
  val kind: String
) {
  sealed class Kind {
    companion object {
      const val GIT = "git"
      const val WORKSPACE = "workspace"
      const val URI = "uri"
    }
  }
}

data class FilepathInGitRepository(
  val filepath: String,
  val gitUrl: String,
  val revision: String? = null,
) : Filepath(Kind.GIT)

data class FilepathInWorkspace(
  val filepath: String,
  val baseDir: String,
) : Filepath(Kind.WORKSPACE)

data class FilepathUri(
  val uri: String,
) : Filepath(Kind.URI)

data class Position(
  // 1-based
  val line: Int,
  val character: Int,
)

abstract class Range
data class LineRange(
  // 1-based
  val start: Int,
  val end: Int,
) : Range()

data class PositionRange(
  val start: Position,
  val end: Position,
) : Range()

data class EditorFileContext(
  val kind: String = "file",
  val filepath: Filepath,
  val range: Range?,
  val content: String,
)

sealed class ChatCommand {
  companion object {
    const val EXPLAIN = "explain"
    const val FIX = "fix"
    const val GENERATE_DOCS = "generate-docs"
    const val GENERATE_TESTS = "generate-tests"
  }
}

data class FileLocation(
  val filepath: Filepath,
  // Int, LineRange, Position, or PositionRange
  val location: Any?,
)

data class GitRepository(
  val url: String,
)

private val gson = Gson()

fun Any.asFileLocation(): FileLocation? {
  val filepath = if (this is Map<*, *> && this.containsKey("filepath")) {
    val filepathValue = this["filepath"]
    if (filepathValue is Map<*, *> && filepathValue.containsKey("kind")) {
      if (filepathValue["kind"] == Filepath.Kind.GIT) {
        gson.fromJson(gson.toJson(filepathValue), FilepathInGitRepository::class.java)
      } else if (filepathValue["kind"] == Filepath.Kind.WORKSPACE) {
        gson.fromJson(gson.toJson(filepathValue), FilepathInWorkspace::class.java)
      } else if (filepathValue["kind"] == Filepath.Kind.URI) {
        gson.fromJson(gson.toJson(filepathValue), FilepathUri::class.java)
      } else {
        null
      }
    } else {
      null
    }
  } else {
    null
  } ?: return null

  val location = if (this is Map<*, *> && containsKey("location")) {
    val locationValue = this["location"]
    if (locationValue is Number) {
      locationValue
    } else if (locationValue is Map<*, *>) {
      if (locationValue.containsKey("line")) {
        gson.fromJson(gson.toJson(locationValue), Position::class.java)
      } else if (locationValue.containsKey("start") && locationValue["start"] is Number) {
        gson.fromJson(gson.toJson(locationValue), LineRange::class.java)
      } else if (locationValue.containsKey("start") && locationValue["start"] is Map<*, *>) {
        gson.fromJson(gson.toJson(locationValue), PositionRange::class.java)
      } else {
        null
      }
    } else {
      null
    }
  } else {
    null
  }

  return FileLocation(filepath, location)
}
