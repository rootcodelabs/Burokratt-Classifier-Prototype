"use client";
import React from "react";
import { Button, Container } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.css";

export default function Home() {
  return (
    <Container fluid>
      <div className="flex flex-col h-screen justify-content-center items-center gap-3">
        <h1>Bürokratt Classifier Prototype</h1>
        <h3>Powered by Rootcode AI</h3>
        <div className="flex flex-row justify-center items-center">
          <img
            src={"https://avatars.githubusercontent.com/u/91940340?s=200&v=4"}
            alt="Company Logo"
            className="border-gray-300"
          />
        </div>
        <Button variant="primary" size="lg" href={`/datasets`} className="w-52">
          Explore
        </Button>
      </div>
    </Container>
  );
}
