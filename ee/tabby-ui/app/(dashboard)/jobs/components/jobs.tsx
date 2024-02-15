import { SubHeader } from "@/components/sub-header"
import { JobRunsTable } from "./jobs-table"

export default function Jobs() {
  return (
    <>
      <SubHeader>
        Job runs
      </SubHeader>
      <JobRunsTable />
    </>
  )

}
