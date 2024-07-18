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

class TabbyBrowser(private val project: Project) {
	private val browser: JBCefBrowser

	init {
		val htmlContent = """
            <!DOCTYPE html>
            <html style="width: 100vw; height: 100vh; margin: 0; padding: 0">
            <head>
                <title>Tabby Browser</title>
            </head>
            <body style="width: 100vw; height: 100vh">
                <iframe src="http://localhost:8080" width="100%" height="100%"></iframe>
                
                <script>
                    window.addEventListener('message', function(event) {
                        var message = event.data;
                         console.log('Received message from plugin:', message);
                    });
                </script>
            </body>
            </html>
        """.trimIndent()

		browser = JBCefBrowser.createBuilder()
			.setOffScreenRendering(false)
			.build()

		browser.loadHTML(htmlContent)

		browser.jbCefClient.addLoadHandler(object : CefLoadHandlerAdapter() {
			override fun onLoadingStateChange(
				browser: CefBrowser?,
				isLoading: Boolean,
				canGoBack: Boolean,
				canGoForward: Boolean
			) {
				if (!isLoading) {
					// The page has finished loading
					println("Page loaded successfully")

					// Post a message to the browser
					sendMessageToBrowser("Hello from the IntelliJ plugin!")
				}
			}

			// ...
		}, browser.cefBrowser)

		Disposer.register(project, browser)


		// Register the browser with the application
		ApplicationManager.getApplication().invokeLater {
			browser.component.apply {
				// Set any additional properties or configurations for the browser component
			}
		}
	}

	fun sendMessageToBrowser(message: String) {
		val jsCode = "window.postMessage('$message', '*');"
		browser.cefBrowser.executeJavaScript(jsCode, null, 0)
	}

	fun getBrowserComponent() = browser.component
}