import React, { useState, useEffect } from "react";
import { marked } from "marked";

const GitHubReadme: React.FC<{
  src?: string;
}> = ({
  src,
}) => {
    if (!src) {
      console.error(
        "react-github-readme-md: You must provide either a src or username and repo"
      );
      return null;
    }

    const [readmeContent, setReadmeContent] = useState<string>("");

    useEffect(() => {
      // Function to fetch the README content from GitHub
      const fetchReadme = async () => {
        try {
          let readmeUrl = "";

          if (src) {
            // Allow passing a URL directly as a prop
            readmeUrl = src;
          }

          if (!readmeUrl) {
            throw new Error("Failed to fetch README path");
          }

          const response = await fetch(readmeUrl);

          if (!response.ok) {
            throw new Error("Failed to fetch README");
          }

          const data = await response.text();

          if (data) {
            setReadmeContent(data.split("\n").splice(1).join("\n"));
          }
        } catch (error) {
          console.error("react-github-readme-md: ", error);
        }
      };

      fetchReadme();
    }, []);

    if (!readmeContent) {
      return null;
    }

    // Parse the markdown content into HTML
    try {
      const ghContent = marked.parse(readmeContent);
      return (
        <div>
          <div
            dangerouslySetInnerHTML={{
              __html: ghContent,
            }}
          />
        </div>
      );
    } catch (error) {
      console.error("react-github-readme-md: ", error);
      return null;
    }
  };

export default GitHubReadme;