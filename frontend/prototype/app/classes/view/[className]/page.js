"use client";
import "bootstrap/dist/css/bootstrap.css";
import React, { useEffect, useState } from "react";
import { Button, Container } from "react-bootstrap";
import { BsArrowLeft } from "react-icons/bs";
import axios from "axios";
import Loader from "@/components/loaders/ScaleLoader";

function ViewClass({ params }) {
  const [isLoading, setIsLoading] = useState(true);
  const Table = ({ data }) => {
    return (
      <div className="lg:h-[480px] overflow-auto">
        <table className="table table-fixed  w-full text-sm">
          <thead>
            <tr className="sticky top-0">
              <th className="px-4 py-2 border  bg-black text-white">Index</th>
              <th className="px-4 py-2 border  bg-black text-white">Data</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={`${index}-${item?.data_id}`}>
                <td className="px-4 py-2 border">{index + 1}</td>
                <td className="px-4 py-2 border">{item?.data}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const [tableData, setTableData] = useState([]);
  const [className, setClassName] = useState(params.className);

  const getDatasetInfo = () => {
    setIsLoading(true);
    axios
      .get(`http://localhost:8000/class/data/${params.className}`)
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
  useEffect(() => {
    getDatasetInfo();
  }, []);

  return (
    <Container fluid>
      <div className="flex flex-col">
        <div className="justify-start mt-2">
          <Button variant="primary" href="/classes" className="w-20" size="lg">
            <BsArrowLeft />
          </Button>
        </div>

        <h2 className="text-center capitalize">
          {decodeURIComponent(className)}
        </h2>
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
export default ViewClass;
