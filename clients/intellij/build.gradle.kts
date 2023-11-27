plugins {
  id("java")
  id("org.jetbrains.kotlin.jvm") version "1.8.21"
  id("org.jetbrains.intellij") version "1.13.3"
  id("org.jetbrains.changelog") version "2.2.0"
}

group = "com.tabbyml"
version = "1.2.0-dev"

repositories {
  mavenCentral()
}

// Configure Gradle IntelliJ Plugin
// Read more: https://plugins.jetbrains.com/docs/intellij/tools-gradle-intellij-plugin.html
intellij {
  version.set("2022.2.5")
  type.set("IC") // Target IDE Platform

  plugins.set(listOf(/* Plugin Dependencies */))
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
    untilBuild.set("233.*")
    changeNotes.set(provider {
      changelog.renderItem(
        changelog.getLatest(),
        org.jetbrains.changelog.Changelog.OutputType.HTML
      )
    })
  }

  val copyNodeScripts by register<Copy>("copyNodeScripts") {
    dependsOn(prepareSandbox)
    from("node_scripts")
    into("build/idea-sandbox/plugins/intellij-tabby/node_scripts")
  }

  buildSearchableOptions {
    dependsOn(copyNodeScripts)
  }

  runIde {
    dependsOn(copyNodeScripts)
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
