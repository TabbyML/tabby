import React from "react"
import ReactDOM from "react-dom/client"
import { StreamlitProvider } from "streamlit-component-lib-react-hooks"
import Monaco from "./Monaco"

const rootElement = document.getElementById("root")
const root = ReactDOM.createRoot(rootElement)

root.render(
  <React.StrictMode>
    <StreamlitProvider>
      <Monaco />
    </StreamlitProvider>
  </React.StrictMode>
)
