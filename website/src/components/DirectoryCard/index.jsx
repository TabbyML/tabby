import React from "react";
import { Card } from "./Card";
import docsData from "../../data/docsData.json"; // Import the generated JSON data

export function DirectoryCards({ currentPath }) {
  console.log("Current Path:", currentPath);

  // Filter documents for the current directory
  const currentDocs = docsData.filter((doc) => {
    if (doc.path.endsWith("/index")) return false; // Exclude index.md
    return doc.path.startsWith(currentPath) && doc.path !== currentPath; // Include only documents in the current path
  });

  console.log("Filtered Docs:", currentDocs);

  // Map filtered documents to cards
  const cards = currentDocs.map((doc) => ({
    title: doc.title,
    description: doc.description || "No description available", // Fallback description
    link: doc.path,
  }));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
      {cards.length > 0 ? (
        cards.map((card, index) => (
          <Card
            key={index}
            title={card.title}
            description={card.description}
            link={card.link}
          />
        ))
      ) : (
        <p className="text-gray-500">
          No documents available in this directory.
        </p>
      )}
    </div>
  );
}
