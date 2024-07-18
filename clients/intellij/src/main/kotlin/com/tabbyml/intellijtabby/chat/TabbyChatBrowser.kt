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

class TabbyBrowser(private val project: Project) {
	private val browser: JBCefBrowser

	init {
		browser = JBCefBrowser.createBuilder()
			.setOffScreenRendering(false)
			.build()

		displayChatPage(browser, "http://localhost:8080")

		val jsQuery = JBCefJSQuery.create(browser)
		jsQuery.addHandler { message ->
			// Handle the message from JavaScript here
			println("Received message: $message")
			JBCefJSQuery.Response("") // Respond back to JS (optional)
		}

		// Inject the window.postMessageToIntellij function into the browser's JavaScript context after the HTML has loaded.
		// This function will be used to send messages from the web page to the IntelliJ client.
		browser.jbCefClient.addLoadHandler(object : CefLoadHandlerAdapter() {
			override fun onLoadingStateChange(
				browser: CefBrowser?,
				isLoading: Boolean,
				canGoBack: Boolean,
				canGoForward: Boolean
			) {
				if (!isLoading) {
					val script = """window.postMessageToIntellij = function(message) {
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
					  		window.postMessageToIntellij(JSON.stringify({ action: 'rendered' }));
					  	}, 300)
					  });
	
					  chatIframe.src=encodeURI("${endpoint}/chat?" + clientQuery)
					}
					
					window.addEventListener("message", (event) => {
					  if (!chatIframe) return
					  if (event.data) {
						
						console.log("window.addEventListener event.data", event.data)
//						if (event.data.data) {
//						  chatIframe.contentWindow.postMessage(event.data.data[0], "${endpoint}");
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

	fun sendMessageToBrowser(message: String) {
		val jsCode = "window.postMessage('$message', '*');"
		browser.cefBrowser.executeJavaScript(jsCode, null, 0)
	}

	fun getBrowserComponent() = browser.component
}