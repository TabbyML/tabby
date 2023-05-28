/* eslint-disable jsx-a11y/anchor-is-valid */
import {
  Breadcrumb,
  Button,
  Checkbox,
  Label,
  Modal,
  Table,
  TextInput,
} from "flowbite-react";
import type { FC } from "react";
import { useState } from "react";
import {
  HiChevronLeft,
  HiChevronRight,
  HiCog,
  HiDocumentDownload,
  HiDotsVertical,
  HiExclamationCircle,
  HiHome,
  HiOutlineExclamationCircle,
  HiOutlinePencilAlt,
  HiPlus,
  HiTrash,
} from "react-icons/hi";
import NavbarSidebarLayout from "../../layouts/navbar-sidebar";

const UserListPage: FC = function () {
  return (
    <NavbarSidebarLayout isFooter={false}>
      <div className="block items-center justify-between border-b border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800 sm:flex">
        <div className="mb-1 w-full">
          <div className="mb-4">
            <Breadcrumb className="mb-4">
              <Breadcrumb.Item href="#">
                <div className="flex items-center gap-x-3">
                  <HiHome className="text-xl" />
                  <span className="dark:text-white">Home</span>
                </div>
              </Breadcrumb.Item>
              <Breadcrumb.Item href="/users/list">Users</Breadcrumb.Item>
              <Breadcrumb.Item>List</Breadcrumb.Item>
            </Breadcrumb>
            <h1 className="text-xl font-semibold text-gray-900 dark:text-white sm:text-2xl">
              All users
            </h1>
          </div>
          <div className="sm:flex">
            <div className="mb-3 hidden items-center dark:divide-gray-700 sm:mb-0 sm:flex sm:divide-x sm:divide-gray-100">
              <form className="lg:pr-3">
                <Label htmlFor="users-search" className="sr-only">
                  Search
                </Label>
                <div className="relative mt-1 lg:w-64 xl:w-96">
                  <TextInput
                    id="users-search"
                    name="users-search"
                    placeholder="Search for users"
                  />
                </div>
              </form>
              <div className="mt-3 flex space-x-1 pl-0 sm:mt-0 sm:pl-2">
                <a
                  href="#"
                  className="inline-flex cursor-pointer justify-center rounded p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"
                >
                  <span className="sr-only">Configure</span>
                  <HiCog className="text-2xl" />
                </a>
                <a
                  href="#"
                  className="inline-flex cursor-pointer justify-center rounded p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"
                >
                  <span className="sr-only">Delete</span>
                  <HiTrash className="text-2xl" />
                </a>
                <a
                  href="#"
                  className="inline-flex cursor-pointer justify-center rounded p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"
                >
                  <span className="sr-only">Purge</span>
                  <HiExclamationCircle className="text-2xl" />
                </a>
                <a
                  href="#"
                  className="inline-flex cursor-pointer justify-center rounded p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"
                >
                  <span className="sr-only">Settings</span>
                  <HiDotsVertical className="text-2xl" />
                </a>
              </div>
            </div>
            <div className="ml-auto flex items-center space-x-2 sm:space-x-3">
              <AddUserModal />
              <Button color="gray">
                <div className="flex items-center gap-x-3">
                  <HiDocumentDownload className="text-xl" />
                  <span>Export</span>
                </div>
              </Button>
            </div>
          </div>
        </div>
      </div>
      <div className="flex flex-col">
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full align-middle">
            <div className="overflow-hidden shadow">
              <AllUsersTable />
            </div>
          </div>
        </div>
      </div>
      <Pagination />
    </NavbarSidebarLayout>
  );
};

const AddUserModal: FC = function () {
  const [isOpen, setOpen] = useState(false);

  return (
    <>
      <Button color="primary" onClick={() => setOpen(true)}>
        <div className="flex items-center gap-x-3">
          <HiPlus className="text-xl" />
          Add user
        </div>
      </Button>
      <Modal onClose={() => setOpen(false)} show={isOpen}>
        <Modal.Header className="border-b border-gray-200 !p-6 dark:border-gray-700">
          <strong>Add new user</strong>
        </Modal.Header>
        <Modal.Body>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <Label htmlFor="firstName">First name</Label>
              <div className="mt-1">
                <TextInput
                  id="firstName"
                  name="firstName"
                  placeholder="Bonnie"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="lastName">Last name</Label>
              <div className="mt-1">
                <TextInput id="lastName" name="lastName" placeholder="Green" />
              </div>
            </div>
            <div>
              <Label htmlFor="email">Email</Label>
              <div className="mt-1">
                <TextInput
                  id="email"
                  name="email"
                  placeholder="example@company.com"
                  type="email"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="phone">Phone number</Label>
              <div className="mt-1">
                <TextInput
                  id="phone"
                  name="phone"
                  placeholder="e.g., +(12)3456 789"
                  type="tel"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="department">Department</Label>
              <div className="mt-1">
                <TextInput
                  id="department"
                  name="department"
                  placeholder="Development"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="company">Company</Label>
              <div className="mt-1">
                <TextInput
                  id="company"
                  name="company"
                  placeholder="Somewhere"
                />
              </div>
            </div>
          </div>
        </Modal.Body>
        <Modal.Footer>
          <Button color="primary" onClick={() => setOpen(false)}>
            Add user
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

const AllUsersTable: FC = function () {
  return (
    <Table className="min-w-full divide-y divide-gray-200 dark:divide-gray-600">
      <Table.Head className="bg-gray-100 dark:bg-gray-700">
        <Table.HeadCell>
          <Label htmlFor="select-all" className="sr-only">
            Select all
          </Label>
          <Checkbox id="select-all" name="select-all" />
        </Table.HeadCell>
        <Table.HeadCell>Name</Table.HeadCell>
        <Table.HeadCell>Position</Table.HeadCell>
        <Table.HeadCell>Country</Table.HeadCell>
        <Table.HeadCell>Status</Table.HeadCell>
        <Table.HeadCell>Actions</Table.HeadCell>
      </Table.Head>
      <Table.Body className="divide-y divide-gray-200 bg-white dark:divide-gray-700 dark:bg-gray-800">
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <Checkbox aria-describedby="checkbox-1" id="checkbox-1" />
              <label htmlFor="checkbox-1" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/neil-sims.png"
              alt="Neil Sims avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Neil Sims
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                neil.sims@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Front-end developer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            United States
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-2"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-2" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/roberta-casas.png"
              alt="Roberta Casas avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Roberta Casas
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                roberta.casas@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Designer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Spain
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-3"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-3" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/michael-gough.png"
              alt="Michael Gough avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Michael Gough
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                michael.gough@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            React developer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            United Kingdom
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-4"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-4" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/jese-leos.png"
              alt="Jese Leos avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Jese Leos
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                jese.leos@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Marketing
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            United States
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-5"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-5" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/bonnie-green.png"
              alt="Bonnie Green avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Bonnie Green
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                bonnie.green@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            UI/UX Engineer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            AusTable.Rowalia
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-red-500"></div>{" "}
              Offline
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-6"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-6" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/thomas-lean.png"
              alt="Thomas Lean avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Thomas Lean
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                thomas.lean@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Vue developer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Germany
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-7"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-7" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/helene-engels.png"
              alt="Helene Engels avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Helene Engels
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                helene.engels@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Product owner
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Canada
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-8"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-8" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/lana-byrd.png"
              alt="Lana Byrd avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Lana Byrd
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                lana.byrd@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Designer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            United States
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-9"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-9" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/leslie-livingston.png"
              alt="Leslie Livingston avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Leslie Livingston
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                leslie.livingston@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Web developer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            France
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-red-500"></div>{" "}
              Offline
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-10"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-10" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/robert-brown.png"
              alt="Robert Brown avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Robert Brown
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                robert.brown@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Laravel developer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Russia
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-11"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-11" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/neil-sims.png"
              alt="Neil Sims avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Neil Sims
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                neil.sims@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Front-end developer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            United States
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-12"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-12" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/roberta-casas.png"
              alt="Roberta Casas avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Roberta Casas
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                roberta.casas@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Designer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Spain
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-13"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-13" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/michael-gough.png"
              alt="Michael Gough avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Michael Gough
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                michael.gough@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            React developer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            United Kingdom
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-14"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-14" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/jese-leos.png"
              alt="Jese Leos avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Jese Leos
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                jese.leos@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Marketing
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            United States
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-15"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-15" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/bonnie-green.png"
              alt="Bonnie Green avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Bonnie Green
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                bonnie.green@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            UI/UX Engineer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            AusTable.Rowalia
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-red-500"></div>{" "}
              Offline
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>

        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-16"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-16" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/thomas-lean.png"
              alt="Thomas Lean avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Thomas Lean
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                thomas.lean@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Vue developer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Germany
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>

        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-17"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-17" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/helene-engels.png"
              alt="Helene Engels avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Helene Engels
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                helene.engels@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Product owner
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Canada
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-18"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-18" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/lana-byrd.png"
              alt="Lana Byrd avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Lana Byrd
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                lana.byrd@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Designer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            United States
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-19"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-19" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/leslie-livingston.png"
              alt="Leslie Livingston avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Leslie Livingston
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                leslie.livingston@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Web developer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            France
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-red-500"></div>{" "}
              Offline
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
        <Table.Row className="hover:bg-gray-100 dark:hover:bg-gray-700">
          <Table.Cell className="w-4 p-4">
            <div className="flex items-center">
              <input
                id="checkbox-20"
                aria-describedby="checkbox-1"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 bg-gray-50 focus:ring-4 focus:ring-primary-300 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-primary-600"
              />
              <label htmlFor="checkbox-20" className="sr-only">
                checkbox
              </label>
            </div>
          </Table.Cell>
          <Table.Cell className="mr-12 flex items-center space-x-6 whitespace-nowrap p-4 lg:mr-0">
            <img
              className="h-10 w-10 rounded-full"
              src="/images/users/robert-brown.png"
              alt="Robert Brown avatar"
            />
            <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
              <div className="text-base font-semibold text-gray-900 dark:text-white">
                Robert Brown
              </div>
              <div className="text-sm font-normal text-gray-500 dark:text-gray-400">
                robert.brown@flowbite.com
              </div>
            </div>
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Laravel developer
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-medium text-gray-900 dark:text-white">
            Russia
          </Table.Cell>
          <Table.Cell className="whitespace-nowrap p-4 text-base font-normal text-gray-900 dark:text-white">
            <div className="flex items-center">
              <div className="mr-2 h-2.5 w-2.5 rounded-full bg-green-400"></div>{" "}
              Active
            </div>
          </Table.Cell>
          <Table.Cell>
            <div className="flex items-center gap-x-3 whitespace-nowrap">
              <EditUserModal />
              <DeleteUserModal />
            </div>
          </Table.Cell>
        </Table.Row>
      </Table.Body>
    </Table>
  );
};

const EditUserModal: FC = function () {
  const [isOpen, setOpen] = useState(false);

  return (
    <>
      <Button color="primary" onClick={() => setOpen(true)}>
        <div className="flex items-center gap-x-2">
          <HiOutlinePencilAlt className="text-lg" />
          Edit user
        </div>
      </Button>
      <Modal onClose={() => setOpen(false)} show={isOpen}>
        <Modal.Header className="border-b border-gray-200 !p-6 dark:border-gray-700">
          <strong>Edit user</strong>
        </Modal.Header>
        <Modal.Body>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <Label htmlFor="firstName">First name</Label>
              <div className="mt-1">
                <TextInput
                  id="firstName"
                  name="firstName"
                  placeholder="Bonnie"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="lastName">Last name</Label>
              <div className="mt-1">
                <TextInput id="lastName" name="lastName" placeholder="Green" />
              </div>
            </div>
            <div>
              <Label htmlFor="email">Email</Label>
              <div className="mt-1">
                <TextInput
                  id="email"
                  name="email"
                  placeholder="example@company.com"
                  type="email"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="phone">Phone number</Label>
              <div className="mt-1">
                <TextInput
                  id="phone"
                  name="phone"
                  placeholder="e.g., +(12)3456 789"
                  type="tel"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="department">Department</Label>
              <div className="mt-1">
                <TextInput
                  id="department"
                  name="department"
                  placeholder="Development"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="company">Company</Label>
              <div className="mt-1">
                <TextInput
                  id="company"
                  name="company"
                  placeholder="Somewhere"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="passwordCurrent">Current password</Label>
              <div className="mt-1">
                <TextInput
                  id="passwordCurrent"
                  name="passwordCurrent"
                  placeholder="••••••••"
                  type="password"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="passwordNew">New password</Label>
              <div className="mt-1">
                <TextInput
                  id="passwordNew"
                  name="passwordNew"
                  placeholder="••••••••"
                  type="password"
                />
              </div>
            </div>
          </div>
        </Modal.Body>
        <Modal.Footer>
          <Button color="primary" onClick={() => setOpen(false)}>
            Save all
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

const DeleteUserModal: FC = function () {
  const [isOpen, setOpen] = useState(false);

  return (
    <>
      <Button color="failure" onClick={() => setOpen(true)}>
        <div className="flex items-center gap-x-2">
          <HiTrash className="text-lg" />
          Delete user
        </div>
      </Button>
      <Modal onClose={() => setOpen(false)} show={isOpen} size="md">
        <Modal.Header className="px-6 pt-6 pb-0">
          <span className="sr-only">Delete user</span>
        </Modal.Header>
        <Modal.Body className="px-6 pt-0 pb-6">
          <div className="flex flex-col items-center gap-y-6 text-center">
            <HiOutlineExclamationCircle className="text-7xl text-red-500" />
            <p className="text-xl text-gray-500">
              Are you sure you want to delete this user?
            </p>
            <div className="flex items-center gap-x-3">
              <Button color="failure" onClick={() => setOpen(false)}>
                Yes, I'm sure
              </Button>
              <Button color="gray" onClick={() => setOpen(false)}>
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
    </>
  );
};

export const Pagination: FC = function () {
  return (
    <div className="sticky right-0 bottom-0 w-full items-center border-t border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800 sm:flex sm:justify-between">
      <div className="mb-4 flex items-center sm:mb-0">
        <a
          href="#"
          className="inline-flex cursor-pointer justify-center rounded p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"
        >
          <span className="sr-only">Previous page</span>
          <HiChevronLeft className="text-2xl" />
        </a>
        <a
          href="#"
          className="mr-2 inline-flex cursor-pointer justify-center rounded p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"
        >
          <span className="sr-only">Next page</span>
          <HiChevronRight className="text-2xl" />
        </a>
        <span className="text-sm font-normal text-gray-500 dark:text-gray-400">
          Showing&nbsp;
          <span className="font-semibold text-gray-900 dark:text-white">
            1-20
          </span>
          &nbsp;of&nbsp;
          <span className="font-semibold text-gray-900 dark:text-white">
            2290
          </span>
        </span>
      </div>
      <div className="flex items-center space-x-3">
        <a
          href="#"
          className="inline-flex flex-1 items-center justify-center rounded-lg bg-primary-700 py-2 px-3 text-center text-sm font-medium text-white hover:bg-primary-800 focus:ring-4 focus:ring-primary-300 dark:bg-primary-600 dark:hover:bg-primary-700 dark:focus:ring-primary-800"
        >
          <HiChevronLeft className="mr-1 text-base" />
          Previous
        </a>
        <a
          href="#"
          className="inline-flex flex-1 items-center justify-center rounded-lg bg-primary-700 py-2 px-3 text-center text-sm font-medium text-white hover:bg-primary-800 focus:ring-4 focus:ring-primary-300 dark:bg-primary-600 dark:hover:bg-primary-700 dark:focus:ring-primary-800"
        >
          Next
          <HiChevronRight className="ml-1 text-base" />
        </a>
      </div>
    </div>
  );
};

export default UserListPage;
