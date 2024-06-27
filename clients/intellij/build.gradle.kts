plugins {
  id("java")
  id("org.jetbrains.kotlin.jvm") version "1.8.21"
  id("org.jetbrains.intellij") version "1.17.4"
  id("org.jetbrains.changelog") version "2.2.0"
}

group = "com.tabbyml"
version = "1.6.0-dev"

repositories {
  mavenCentral()
}

dependencies {
  implementation("org.eclipse.lsp4j:org.eclipse.lsp4j:0.23.1")
}

// Configure Gradle IntelliJ Plugin
// Read more: https://plugins.jetbrains.com/docs/intellij/tools-gradle-intellij-plugin.html
intellij {
  version.set("2024.1")
  type.set("IC") // Target IDE Platform
  plugins.set(listOf("Git4Idea"))
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

  patchPluginXml {
    sinceBuild.set("222")
    changeNotes.set(provider {
      changelog.renderItem(
        changelog.getLatest(),
        org.jetbrains.changelog.Changelog.OutputType.HTML
      )
    })
  }

  register("buildAgent") {
    exec {
      commandLine("pnpm", "turbo", "build")
    }
  }

  prepareSandbox {
    dependsOn("buildAgent")
    from(
      fileTree("node_modules/tabby-agent/dist/") {
        include("node/**/*")
        exclude("**/*.js.map")
      }
    ) {
      into("intellij-tabby/tabby-agent/")
    }
  }

  signPlugin {
    certificateChain.set(System.getenv("CERTIFICATE_CHAIN"))
    privateKey.set(System.getenv("PRIVATE_KEY"))
    password.set(System.getenv("PRIVATE_KEY_PASSWORD"))
  }

  publishPlugin {
    token.set(System.getenv("PUBLISH_TOKEN"))
    channels.set(listOf("alpha"))
  }
}

