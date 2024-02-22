"use client";
import "bootstrap/dist/css/bootstrap.css";
import React, { useEffect, useState } from "react";
import { Button, Container } from "react-bootstrap";
import { BsArrowLeft } from "react-icons/bs";
import { useSearchParams } from "next/navigation";
import axios from "axios";
import Loader from "@/components/loaders/ScaleLoader";

function Dataset({ params }) {
  const [isLoading, setIsLoading] = useState(true);
  const Table = ({ data }) => {
    return (
      <div className="lg:h-[480px] overflow-auto">
        <table className="table table-fixed  w-full text-sm">
          <thead>
            <tr className="sticky top-0">
              <th className="px-4 py-2 border  bg-black text-white">Data ID</th>
              <th className="px-4 py-2 border  bg-black text-white">Data</th>
              <th className="px-4 py-2 border  bg-black text-white">Label</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item) => (
              <tr key={item?.data_id}>
                <td className="px-4 py-2 border">{item?.data_id}</td>
                <td className="px-4 py-2 border">{item?.data}</td>
                <td className="px-4 py-2 border">{item?.class_name}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const search = useSearchParams();
  const [tableData, setTableData] = useState([]); // Manage table data state
  const [dataSetName, setDatasetName] = useState(search.get("name"));
  const [dataSetId, setDatasetID] = useState(params.datasetId);

  useEffect(() => {
    const getDatasetInfo = () => {
      setIsLoading(true);
      axios
        .get(`http://backend:8000/datasets/${params.datasetId}`)
        .then((response) => {
          // Process the response
          setTableData(response?.data?.data);
          setIsLoading(false);
        })
        .catch((error) => {
          // Handle errors
          console.error("Error fetching data:", error);
          alert("Error!");
          setIsLoading(false);
        });
    };
    getDatasetInfo();
  }, []);

  return (
    <Container fluid>
      <div className="flex flex-col">
        <div className="justify-start mt-2">
          <Button variant="primary" href="/datasets" className="w-20" size="lg">
            <BsArrowLeft />
          </Button>
        </div>

        <h2 className="text-center capitalize">{dataSetName}</h2>

        {isLoading ? (
          <Loader />
        ) : (
          <>
            <div className="px-20 mt-10 mb-10 ">
              <Table data={tableData} />
            </div>
          </>
        )}
      </div>
    </Container>
  );
}
export default Dataset;
