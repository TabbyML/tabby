// Configure Gradle IntelliJ Plugin
// Read more: https://plugins.jetbrains.com/docs/intellij/tools-gradle-intellij-plugin.html

plugins {
  id("java")
  id("org.jetbrains.kotlin.jvm") version "1.9.25"
  id("org.jetbrains.intellij.platform") version "2.0.0"
  id("org.jetbrains.changelog") version "2.2.0"
}

repositories {
  mavenCentral()
  intellijPlatform {
    defaultRepositories()
  }
}

dependencies {
  intellijPlatform {
    intellijIdeaCommunity("2023.1")
    bundledPlugins(
      listOf(
        "Git4Idea",
        "org.jetbrains.kotlin",
      )
    )
    pluginVerifier()
    zipSigner()
    instrumentationTools()
  }
  implementation("org.eclipse.lsp4j:org.eclipse.lsp4j:0.23.1")
  implementation("io.github.z4kn4fein:semver:2.0.0")
}

tasks {
  // Set the JVM compatibility versions
  withType<JavaCompile> {
    sourceCompatibility = "17"
    targetCompatibility = "17"
  }
  withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions.jvmTarget = "17"
  }

  intellijPlatform {
    pluginConfiguration {
      version.set("1.11.0-dev")
      changeNotes.set(provider {
        changelog.renderItem(
          changelog.getLatest(),
          org.jetbrains.changelog.Changelog.OutputType.HTML
        )
      })
      ideaVersion {
        sinceBuild.set("231")
        untilBuild.set(provider { null })
      }
    }
    pluginVerification {
      ides {
        recommended()
      }
    }
    signing {
      certificateChain.set(System.getenv("CERTIFICATE_CHAIN"))
      privateKey.set(System.getenv("PRIVATE_KEY"))
    }
    publishing {
      token.set(System.getenv("PUBLISH_TOKEN"))
      channels.set(listOf(System.getenv("PUBLISH_CHANNEL")))
    }
  }

  register("buildDependencies") {
    exec {
      commandLine("pnpm", "turbo", "build")
    }
  }

  prepareSandbox {
    dependsOn("buildDependencies")

    // Copy the tabby-agent to the sandbox
    from(
      fileTree("node_modules/tabby-agent/dist/") {
        include("node/**/*")
        exclude("**/*.js.map")
      }
    ) {
      into("intellij-tabby/tabby-agent/")
    }

    // Copy the tabby-threads `create-thread-from-iframe` to the sandbox
    from(
      fileTree("node_modules/tabby-threads/dist/") {
        include("iife/create-thread-from-iframe.js")
      }
    ) {
      into("intellij-tabby/tabby-threads/")
    }
  }
}

