"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { MdOutlineDatasetLinked } from "react-icons/md";
import { MdOutlineClass } from "react-icons/md";
import { LuBrainCircuit } from "react-icons/lu";
import { SiSpeedtest } from "react-icons/si";

// import logo from "@/img/logo.svg";

export default function Sidebar({ show, setter }) {
  const pathname = usePathname();

  // Define our base class
  const className =
    "bg-gray-200 w-[250px] transition-[margin-left] ease-in-out duration-500 fixed md:static top-0 bottom-0 left-0 z-40";
  // Append class based on state of sidebar visiblity
  const appendClass = show ? " ml-0" : " ml-[-250px] md:ml-0";

  // Clickable menu items
  const MenuItem = ({ icon, name, route }) => {
    // Highlight menu item based on currently displayed route

    const slicedPathName = pathname.split("/").slice(0, 2).join("/");

    const colorClass =
      slicedPathName === route
        ? "text-black"
        : "text-blue-600 hover:text-blue-800";

    return (
      <Link
        href={route}
        onClick={() => {
          setter((oldVal) => !oldVal);
        }}
        className={`flex gap-1 [&>*]:my-auto text-md pl-6 py-3 border-b-[1px] border-b-white/10 ${colorClass}`}
        style={{ textDecoration: "none" }}
      >
        <div className="text-xl flex [&>*]:mx-auto w-[30px]">{icon}</div>
        <div>{name}</div>
      </Link>
    );
  };

  // Overlay to prevent clicks in background, also serves as our close button
  const ModalOverlay = () => (
    <div
      className={`flex md:hidden fixed top-0 right-0 bottom-0 left-0 bg-black/50 z-30`}
      onClick={() => {
        setter((oldVal) => !oldVal);
      }}
    />
  );

  return (
    <>
      <div className={`${className}${appendClass}`}>
        <div className="p-2 flex">
          <Link href="/">
            {/*eslint-disable-next-line*/}
            <img
              src={"https://avatars.githubusercontent.com/u/91940340?s=200&v=4"}
              alt="Company Logo"
              width={300}
              height={300}
              className="border-gray-300"
            />
          </Link>
        </div>
        <div className="flex flex-col font-bold">
          <MenuItem
            name="Datasets"
            route="/datasets"
            icon={<MdOutlineDatasetLinked />}
          />
          <MenuItem name="Classes" route="/classes" icon={<MdOutlineClass />} />
          <MenuItem name="Models" route="/models" icon={<LuBrainCircuit />} />
          <MenuItem
            name="Test"
            route="/try_classifier"
            icon={<SiSpeedtest />}
          />
        </div>
        {/* Add the text "Powered by Rootcode AI" here */}
        <div className="mt-auto absolute bottom-0 p-3 text-center text-sm font-bold text-black">
          Powered by Rootcode AI
        </div>
      </div>
      {show ? <ModalOverlay /> : <></>}
    </>
  );
}
