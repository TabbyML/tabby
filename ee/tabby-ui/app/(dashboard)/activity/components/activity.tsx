import {
  PlusCircle,
} from "lucide-react"


import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardFooter,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import moment from "moment"

export default function Activity() {
  const data = [{
    type: 'completion',
    user: 'bob@tabbyml.com',
    date: moment().subtract(1, 'second').toDate()
  }, {
    type: 'select',
    user: 'alice@tabbyml.com',
    date: moment().subtract(10, 'second').toDate()
  }, {
    type: 'completion',
    user: 'bob@tabbyml.com',
    date: moment().subtract(10, 'hour').toDate()
  }, {
    type: 'views',
    user: 'james@tabbyml.com',
    date: moment().subtract(25, 'hour').toDate()
  }, {
    type: 'select',
    user: 'kevin@tabbyml.com',
    date: moment().subtract(30, 'hour').toDate()
  }]

  return (
    <div className="flex min-h-screen w-full flex-col">
      <div className="flex flex-col sm:gap-4 sm:py-4 sm:pl-14">
        <main className="grid flex-1 items-start gap-4 p-4 sm:px-6 sm:py-0">
          <div className="ml-auto flex items-center gap-2">
            <Button size="sm" className="h-8 gap-1">
              <PlusCircle className="h-3.5 w-3.5" />
              <span className="sr-only sm:not-sr-only sm:whitespace-nowrap">
                Add Product
              </span>
            </Button>
          </div>

          <Card x-chunk="dashboard-06-chunk-0" className="bg-transparent">
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Event</TableHead>
                    <TableHead>People</TableHead>
                    <TableHead>Time</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.map((item, idx) => {
                    return (
                      <TableRow>
                        <TableCell className="font-medium">
                          {item.type}
                        </TableCell>
                        <TableCell>
                          {item.user}
                        </TableCell>
                        <TableCell>
                          {moment(item.date).isBefore(moment().subtract(1, 'days')) ? moment(item.date).format('YYYY-MM-DD HH:mm') : moment(item.date).fromNow()}
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </CardContent>
            <CardFooter>
              <div className="text-xs text-muted-foreground">
                Showing <strong>1-10</strong> of <strong>32</strong>{" "}
                products
              </div>
            </CardFooter>
          </Card>
        </main>
      </div>
    </div>
  )
}
