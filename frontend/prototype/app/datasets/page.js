"use client";
import React, { useEffect, useState } from "react";
import { Button, Container } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.css";
import { IoIosAddCircleOutline } from "react-icons/io";
import { IoDocumentTextOutline } from "react-icons/io5";
import Link from "next/link";
import UploadDatasetModal from "@/components/modals/UploadDatasetModal";
import axios from "axios";
import Loader from "@/components/loaders/ScaleLoader";

export default function Datasets() {
  const [dataSets, setDataSets] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  const [showUploadDatasetModal, setShowUploadDatasetModal] = useState(false);

  const getDatasetsInfo = () => {
    setIsLoading(true);
    axios
      .get("http://backend:8000/datasets/info/")
      .then((response) => {
        setDataSets(response?.data);
        setIsLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        alert("Error!");
        setIsLoading(false);
      });
  };

  useEffect(() => {
    getDatasetsInfo();
  }, []);

  return (
    <Container fluid>
      <div className="lg:px-80 mt-5">
        <div className="flex flex-col">
          <div className="d-flex justify-content-center flex-column mb-5">
            <Button
              variant="primary"
              size="lg"
              className="my-4"
              onClick={() => setShowUploadDatasetModal(true)}
            >
              <div className="flex flex-col gap-2 m-2 justify-center items-center">
                <div className="justify-center">
                  <IoIosAddCircleOutline className="w-10 h-10" />
                </div>
                <div>Upload Dataset</div>
                <div>JSON/CSV/YAML</div>
              </div>
            </Button>
          </div>
          <div>
            <h3>Datasets</h3>
            <div className="flex flex-col gap-3 capitalize max-h-[300px] overflow-y-auto m-4">
              {isLoading ? (
                <Loader />
              ) : (
                <>
                  {" "}
                  {dataSets.map((dataset) => (
                    <Link
                      key={dataset.dataset_id}
                      href={{
                        pathname: `/datasets/${dataset.dataset_id}`,
                        query: { name: dataset.dataset_name },
                      }}
                      style={{ textDecoration: "none" }}
                    >
                      <div
                        key={dataset.dataset_id}
                        className="px-5 flex flex-row items-center gap-1.5"
                      >
                        <div>
                          <IoDocumentTextOutline
                            color="#0D6FED"
                            className="w-8 h-8"
                          />
                        </div>
                        <div className="text-lg">{dataset.dataset_name}</div>
                      </div>
                    </Link>
                  ))}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
      <UploadDatasetModal
        show={showUploadDatasetModal}
        onHide={() => setShowUploadDatasetModal(false)}
        onDatasetUploaded={getDatasetsInfo}
      />
    </Container>
  );
}
