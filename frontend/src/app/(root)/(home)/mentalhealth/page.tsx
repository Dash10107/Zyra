"use client"; // Add this line at the top

import axios from "axios";
import React, { useState } from "react";

const Ask: React.FC = () => {
  const [question, setQuestion] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");

  const formatToOrderedList = (input: string): string => {
    const lines = input.split("\n").filter((line) => line.trim() !== "");
    let orderedList = "<ol class='list-decimal space-y-2 pl-5 text-left'>";

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.startsWith("* **")) {
        const content = line.replace("* **", "").replace(":**", ":");
        orderedList += `
          <li class='flex items-start gap-2'>
            <span class='text-blue-400 font-semibold'>🔹</span>
            <div class='text-blue-400 font-semibold'>${content}</div>
          </li>`;
      } else if (line.startsWith("* ")) {
        const content = line.replace("* ", "");
        orderedList += `
          <li class='flex items-start gap-2'>
            <span class='text-gray-400'>•</span>
            <div class='text-gray-200'>${content}</div>
          </li>`;
      } else {
        orderedList += `
          <p class='text-gray-300 bg-gray-700 p-2 rounded-md'>
            ${line}
          </p>`;
      }
    }

    orderedList += "</ol>";
    return orderedList;
  };

  const handleAsk = async () => {
    if (!question.trim()) {
      setAnswer("<p class='text-red-500'>Please enter a valid question.</p>");
      return;
    }

    const ques = { question };
    const response = await axios.post(
      `https://hqxcjph1-5000.inc1.devtunnels.ms/mentalchat`,
      ques
    );

    if (response) {
      setAnswer("Thinking...");
      const formatted = formatToOrderedList(response.data?.response);
      setAnswer(formatted);
      console.log("response: ", response.data?.response);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-blue-900 via-gray-900 to-black text-white">
      <div className="w-full max-w-md p-6 bg-[#1e293b] shadow-xl rounded-lg">
        <h1 className="text-2xl font-bold text-center mb-4">Ask Anything</h1>
        <p className="text-gray-300 text-sm text-center mb-6">
          Type your question below and get a detailed answer.
        </p>
        <div className="space-y-4">
          <input
            type="text"
            className="w-full px-4 py-2 bg-gray-800 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Type your question here..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button
            onClick={handleAsk}
            className="w-full py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-200 shadow-md"
          >
            Submit
          </button>
        </div>
      </div>
      {answer && (
        <div className="w-full max-w-2xl mt-6 bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4 text-blue-400 border-b border-gray-600 pb-2">
            Answer:
          </h2>
          <div
            className="text-gray-200 leading-relaxed space-y-3"
            dangerouslySetInnerHTML={{ __html: answer }}
          ></div>
        </div>
      )}
    </div>
  );
};

export default Ask;
