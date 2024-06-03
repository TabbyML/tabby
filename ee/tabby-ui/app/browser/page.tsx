"use client";

import { useEffect, useState } from "react";

export default function Page() {
    const [path, setPath] = useState("");
    useEffect(() => {
        const url = new URL(window.location.href);
        setPath(url.pathname);
    }, []);
    return (
      <div className="flex flex-col">
        <p>{path}</p>
      </div>
    )
  }
  