import React from "react";
import Link from "@docusaurus/Link";
// TODO: use custom styles color
export function Card({ title, description, link }) {
  return (
    <Link to={link} className="no-underline block">
      <div
        style={{
          backgroundColor: "rgb(243, 236, 222)",
          // "--hover-bg": "rgb(242, 235, 223)",
        }}
        className="p-6 rounded-lg border border-stone-200 shadow-sm hover:shadow-lg hover:-translate-y-1 hover:border-stone-300 transition-all duration-200 ease-in-out group hover:[background-color:var(--hover-bg)]"
      >
        <div className="flex items-center gap-2 mb-3">
          <h3 className="text-lg font-semibold m-0 text-stone-800 group-hover:text-stone-900 transition-colors">
            {title}
          </h3>
        </div>
        <p
          className="text-stone-600 m-0 text-sm leading-relaxed"
          style={{
            overflow: "hidden",
            whiteSpace: "nowrap",
            textOverflow: "ellipsis",
          }}
        >
          {description}
        </p>
      </div>
    </Link>
  );
}
