import React from "react";

import { ScaleLoader } from "react-spinners";

function Loader() {
  return (
    <div className="flex flex-col mt-20 justify-center items-center ">
      <ScaleLoader  color="#0d6efd"  height={55} width={10} radius={20}/>
    </div>
  );
}

export default Loader;
