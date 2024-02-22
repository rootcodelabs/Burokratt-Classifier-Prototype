"use client";
import React, { useEffect, useState } from "react";
import { Button, Container } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.css";
import { getAllBaseModelsAPIData } from "@/Temp/data";
import CreateModelModal from "@/components/modals/CreateModelModal";
import ModelCard from "@/components/cards/ModelCard";
import axios from "axios";
import TrainingProgressBar from "@/components/loaders/TrainingProgressBar";
import Loader from "@/components/loaders/ScaleLoader";

export default function Models() {
  const [allClasses, setAllClasses] = useState([]); // Manage table data state
  const [allModels, setAllModels] = useState([]);
  const [baseModels, setBaseModels] = useState(getAllBaseModelsAPIData);

  const [showCreateModelModal, setShowCreateModelModal] = useState(false);
  const [isTrainingLoading, setIsTrainingLoading] = useState(false);
  const [isDataLoading, setIsDataLoading] = useState(true);

  const getAllModelsInfo = () => {
    setIsDataLoading(true);
    axios
      .get(`http://backend:8000/model_info/`)
      .then((response) => {
        setAllModels(response.data);
        setIsDataLoading(false);
      })
      .catch((error) => {
        // Handle errors
        console.error("Error fetching data:", error);
        alert("Error!");
        setIsDataLoading(false);
      });
  };

  const getAllClassesInfo = () => {
    setIsDataLoading(true);
    axios
      .get(`http://backend:8000/class_names/`)
      .then((response) => {
        // Process the response
        setAllClasses(response?.data?.data);
        setIsDataLoading(false);
      })
      .catch((error) => {
        // Handle errors
        console.error("Error fetching data:", error);
        setIsDataLoading(false);
      });
  };
  useEffect(() => {
    getAllModelsInfo();
    getAllClassesInfo();
  }, []);

  const handleCreateModelClick = () => {
    getAllClassesInfo();
    setShowCreateModelModal(true);
  };

  const createModel = (modelData) => {
    setIsTrainingLoading(true);
    axios
      .post(`http://backend:8000/train_and_evaluate/`, modelData)
      .then((response) => {
        if (response?.data?.Success) {
          alert("Model Created Successfully!");
          setIsTrainingLoading(false);
          getAllModelsInfo();
        } else {
          alert("Error!");
          setIsTrainingLoading(false);
        }
      })
      .catch((error) => {
        // Handle errors
        console.error("Error fetching data:", error);
        alert("Error!");
        setIsTrainingLoading(false);
      });
  };

  return (
    <>
      {isTrainingLoading ? (
        <TrainingProgressBar />
      ) : (
        <Container fluid>
          <div className="lg:px-80 mt-5">
            <div className="flex flex-col">
              <div className="d-flex justify-content-center flex-column mb-5">
                <Button
                  variant="primary"
                  size="lg"
                  className="my-4 h-20"
                  onClick={handleCreateModelClick}
                >
                  <div className="flex flex-col gap-2 m-2 justify-center items-center">
                    <div>Create Model</div>
                  </div>
                </Button>
              </div>
              <div>
                <h4>Models</h4>

                <div className="flex flex-col max-h-[300px] overflow-y-auto m-4">
                  {isDataLoading ? (
                    <Loader />
                  ) : (
                    <>
                      {allModels?.map((model) => (
                        <div key={model?.model_id}>
                          <div>
                            <ModelCard key={model?.model_id} model={model} />
                          </div>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
          <CreateModelModal
            show={showCreateModelModal}
            onHide={() => setShowCreateModelModal(false)}
            baseModels={baseModels}
            classes={allClasses}
            createModelCallback={createModel}
          />
        </Container>
      )}
    </>
  );
}
