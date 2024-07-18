package com.tabbyml.intellijtabby.chat

import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.Disposer
import com.intellij.ui.jcef.JBCefBrowser
import org.cef.browser.CefBrowser
import org.cef.handler.CefLoadHandlerAdapter
import org.cef.callback.CefCallback
import org.cef.handler.CefLoadHandler
import org.cef.network.CefRequest
import com.intellij.ui.jcef.JBCefJSQuery
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.util.UUID

class TabbyBrowser(private val project: Project) {
	private val browser: JBCefBrowser

	init {
		browser = JBCefBrowser.createBuilder()
			.setOffScreenRendering(false)
			.build()

		// FIXME: how to get the endpoint
		displayChatPage(browser, "http://localhost:8080")

		// Listen to the message sent from the web page
		val jsQuery = JBCefJSQuery.create(browser)
		jsQuery.addHandler { message: String ->
			val parser = JsonParser()
			val json: JsonObject = parser.parse(message).asJsonObject

			val action = json.get("action")?.asString
			if (action == "rendered") {
				// FIXME: how to get the server token
				sendMessageToServer("init", listOf(mapOf("fetcherOptions" to mapOf("authorization" to "auth_95e5a81225a84b9f8c3ae1659c890856"))))
				return@addHandler  JBCefJSQuery.Response("")
			}

			JBCefJSQuery.Response("")
		}

		// Inject the window.onReceiveMessage function into the browser's JavaScript context after the HTML has loaded
		// This function will be used to send messages from the web page to the IntelliJ client
		browser.jbCefClient.addLoadHandler(object : CefLoadHandlerAdapter() {
			override fun onLoadingStateChange(
				browser: CefBrowser?,
				isLoading: Boolean,
				canGoBack: Boolean,
				canGoForward: Boolean
			) {
				if (!isLoading) {
					val script = """window.onReceiveMessage = function(message) {
						${jsQuery.inject("message")}
					}""".trimIndent()
					browser?.executeJavaScript(
						script,
						browser.url,
						0
					)
				}
			}
		}, browser.cefBrowser)

		Disposer.register(project, browser)
	}

	fun displayChatPage(browser: JBCefBrowser, endpoint: String) {
		val htmlContent = """
            <!DOCTYPE html>
            <html  lang="en" style="width: 100vw; height: 100vh; margin: 0; padding: 0">
            <head>
                <meta charset="UTF-8" />
            	<meta name="viewport" content="width=device-width, initial-scale=1.0" />
			
				<script defer>
				  window.onload = function () {
					const chatIframe = document.getElementById("chat");
					if (chatIframe) {
					  const clientQuery = "&client=intellij"
		  
					  chatIframe.addEventListener('load', function() {
					  	setTimeout(() => {
					  		window.onReceiveMessage(JSON.stringify({ action: 'rendered' }));
					  	}, 1000)
					  });
	
					  chatIframe.src=encodeURI("${endpoint}/chat?" + clientQuery)
					}
					
					window.addEventListener("message", (event) => {
					  if (!chatIframe) return
					  if (event.data) {
						 if (event.data === "quilt.threads.pong") return // @quilted/threads message
						 
						 
						 // FIXME: how to distinguish the message from client to html and from html to client
						 
						console.log("window.addEventListener event.data", event.data)
						
						chatIframe.contentWindow.postMessage(JSON.parse(event.data), "${endpoint}");
						
//						if (event.data.data) {
//						  chatIframe.contentWindow.postMessage(event.data, "${endpoint}");
//						} else {
//						  vscode.postMessage(event.data);
//						}
					  }
					});
				  }
				</script>
            </head>
            <body style="width: 100vw; height: 100vh">
                <iframe
					id="chat"
              		allow="clipboard-read; clipboard-write"
			   		style="height: 100%; width: 100%" />
            </body>
            </html>
        """.trimIndent()
		browser.loadHTML(htmlContent)
	}

	fun sendMessageToChat(message: Any) {
		val gson = Gson()
		val jsonString = gson.toJson(message)
		val jsCode = "window.postMessage('$jsonString', '*');"
		browser.cefBrowser.executeJavaScript(jsCode, null, 0)
	}

	fun getBrowserComponent() = browser.component

	// FIXME: refactory into a class
	fun sendMessageToServer(methodName: String, params: List<Any?>) {
		val uuid = UUID.randomUUID()
		val threadMessage = listOf(
			0,
			listOf(
				uuid.toString(),
				methodName,
				params
			)
		)
		sendMessageToChat(threadMessage)
	}
}