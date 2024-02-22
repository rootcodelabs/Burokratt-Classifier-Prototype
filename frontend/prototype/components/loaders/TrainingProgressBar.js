import "bootstrap/dist/css/bootstrap.css";
import React from "react";

import { RingLoader } from "react-spinners";

function TrainingProgressBar() {
  return (
    <div className="flex flex-col h-screen justify-content-center items-center gap-5">
      <h2>Training in Progress</h2>
      <RingLoader color="#0d6efd" />
      <h4>This might take few minutes</h4>
    </div>
  );
}

export default TrainingProgressBar;
