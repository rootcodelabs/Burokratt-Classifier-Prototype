"use client";
import "bootstrap/dist/css/bootstrap.css";
import { Button } from "react-bootstrap";
import UpdateClassModal from "../modals/UpdateClassModal";
import { useState } from "react";
import axios from "axios";

export default function ClassCard({ item, onUpdateSuccessCallback }) {
  const [showUpdateClassModal, setShowUpdateClassModal] = useState(false);

  const handleDeleteClass = (className) => {
    axios
      .delete(`http://backend:8000/class/${className}`)
      .then((response) => {
        if (response.status == 200) {
          alert("Class Deleted Successfully!");
          onUpdateSuccessCallback();
        }
      })
      .catch((error) => {
        // Handle errors
        console.error("Error fetching data:", error);
      });
  };

  return (
    <div className="bg-gray-100 p-4 my-3 rounded shadow mx-3">
      <div className="flex flex-row items-center gap-3 cursor-pointer">
        <h3 className="basis-1/2 text-xl font-bold">{item}</h3>
        <Button
          size="sm"
          className="w-20"
          variant="primary"
          href={`/classes/view/${item}`}
        >
          View
        </Button>
        <Button
          size="sm"
          className="w-20"
          variant="success"
          onClick={() => {
            setShowUpdateClassModal(true);
          }}
        >
          Update
        </Button>
        <Button
          size="sm"
          className="w-20"
          variant="danger"
          onClick={() => {
            handleDeleteClass(item);
          }}
        >
          Delete
        </Button>
      </div>
      <UpdateClassModal
        show={showUpdateClassModal}
        onHide={() => setShowUpdateClassModal(false)}
        className={item}
        onSuccessCallBack={onUpdateSuccessCallback}
      />
    </div>
  );
}
