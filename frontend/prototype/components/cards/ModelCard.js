"use client";
import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.css";

export default function ModelCard({ model }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="bg-gray-100 p-4 my-4 rounded shadow mx-3">
      <div
        className="flex justify-between items-center cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h3 className="text-xl font-bold">{model?.model_name}</h3>
        <div className="font-bold text-sm">{`Accuracy : ${parseFloat(
          model?.model_accuracy
        ).toFixed(2)}`}</div>
        <div className="font-bold text-sm">{`F1 Score : ${parseFloat(
          model?.model_f1
        ).toFixed(2)}`}</div>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          className={`w-6 h-6 transition-transform transform ${
            isExpanded ? "rotate-180" : "rotate-0"
          }`}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </div>
      {isExpanded && (
        <div className="mt-4">
          <table className="w-full border border-gray-300">
            <thead>
              <tr className="bg-gray-200">
                <th className="py-2 px-4">Class Name</th>
                <th className="py-2 px-4">F1 Score</th>
                <th className="py-2 px-4">Precision</th>
              </tr>
            </thead>
            <tbody>
              {model?.scores?.map((row, index) => (
                <tr key={`${model?.model_id}-${index}`}>
                  <td className="py-2 px-4 border">{row?.class_name}</td>
                  <td className="py-2 px-4 border">{row?.f1_score}</td>
                  <td className="py-2 px-4 border">{row?.precision}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
