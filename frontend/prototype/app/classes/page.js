"use client";
import React, { useEffect, useState } from "react";
import { Button, Container } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.css";
import CreateClassModal from "@/components/modals/CreateClassModal";
import { getAllDatasetsAPIData, getSingleClassAPIData } from "@/Temp/data";
import axios from "axios";
import ClassCard from "@/components/cards/ClassCard";
import Loader from "@/components/loaders/ScaleLoader";

export default function Classes() {
  const [allClasses, setAllClasses] = useState([]);
  const [classData, setClassData] = useState(getSingleClassAPIData);
  const [allDatasets, setAllDatasets] = useState(getAllDatasetsAPIData);

  const [showCreateClassModal, setShowCreateClassModal] = useState(false);

  const [isLoading, setIsLoading] = useState(false);

  const getClassesInfo = () => {
    setIsLoading(true);
    axios
      .get(`http://0.0.0.0:8000/class_names/`)
      .then((response) => {
        // Process the response
        setAllClasses(response?.data?.data);
        setIsLoading(false);
      })
      .catch((error) => {
        // Handle errors
        console.error("Error fetching data:", error);
        setIsLoading(false);
      });
  };
  useEffect(() => {
    getClassesInfo();
  }, []);

  const handleCreateClassClick = () => {
    setShowCreateClassModal(true);
  };

  return (
    <Container fluid>
      <div className="lg:px-80 mt-5">
        <div className="flex flex-col">
          <div className="d-flex justify-content-center flex-column mb-5">
            <Button
              variant="primary"
              size="lg"
              className="my-4 h-20"
              onClick={handleCreateClassClick}
            >
              <div className="flex flex-col gap-2 m-2 justify-center items-center">
                <div>Create Class</div>
              </div>
            </Button>
          </div>
          <div>
            <h4>Classes</h4>
            <div className="flex flex-col gap-2 capitalize max-h-[350px] overflow-y-auto m-4">
              {isLoading ? (
                <Loader />
              ) : (
                <>
                  {allClasses?.map((singleClass, index) => {
                    return (
                      <div key={index}>
                        <ClassCard
                          key={index}
                          item={singleClass}
                          onUpdateSuccessCallback={getClassesInfo}
                        />
                      </div>
                    );
                  })}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
      <CreateClassModal
        show={showCreateClassModal}
        onHide={() => setShowCreateClassModal(false)}
        datasets={allDatasets}
        onSuccessCallBack={getClassesInfo}
      />
    </Container>
  );
}
