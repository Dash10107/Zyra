import { SettingsIcon } from "lucide-react";
import { Button } from "./ui/button";
import DottedSeparator from "./dotted-separator";
import Link from "next/link";
import { useWorkspaceId } from "@/features/workspaces/hook/use-workspace-id";
import { Card, CardContent } from "./ui/card";
import { Member } from "@/features/members/type";
import { MemberAvatar } from "@/features/members/components/member-avatar";

interface MemberListProps {
  data: Member[];
  total: number;
}

const membersData = [
  {
    name: "Alice",
    location: "New York",
    timezone: "EST/EDT",
    working_hours: "9:00 AM - 5:00 PM",
    gmt_offset: "GMT-4/-5",
    best_meeting_time_options: [
      {
        option: "1:00 PM - 2:00 PM GMT",
        local_time: "9:00 AM - 10:00 AM EDT",
        explanation: "This is within Alice's core working hours, making it convenient."
      },
      {
        option: "2:00 PM - 3:00 PM GMT",
        local_time: "10:00 AM - 11:00 AM EDT",
        explanation: "This is still within Alice's core working hours and allows for a later start."
      },
      {
        option: "12:00 PM - 1:00 PM GMT",
        local_time: "8:00 AM - 9:00 AM EDT",
        explanation: "This is early in Alice's working hours but still feasible."
      }
    ]
  },
  {
    name: "Bob",
    location: "London",
    timezone: "GMT/BST",
    working_hours: "9:00 AM - 5:00 PM",
    gmt_offset: "GMT/GMT+1",
    best_meeting_time_options: [
      {
        option: "1:00 PM - 2:00 PM GMT",
        local_time: "2:00 PM - 3:00 PM BST",
        explanation: "This time falls comfortably within Bob's working hours."
      },
      {
        option: "2:00 PM - 3:00 PM GMT",
        local_time: "3:00 PM - 4:00 PM BST",
        explanation: "This is still within Bob's working hours but closer to the end of the day."
      },
      {
        option: "12:00 PM - 1:00 PM GMT",
        local_time: "1:00 PM - 2:00 PM BST",
        explanation: "This time is also within Bob's working hours and is convenient."
      }
    ]
  },
  {
    name: "Charlie",
    location: "Tokyo",
    timezone: "JST",
    working_hours: "9:00 AM - 5:00 PM",
    gmt_offset: "GMT+9",
    best_meeting_time_options: [
      {
        option: "1:00 PM - 2:00 PM GMT",
        local_time: "10:00 PM - 11:00 PM JST",
        explanation: "This is late but manageable for Charlie if flexible hours are possible."
      },
      {
        option: "2:00 PM - 3:00 PM GMT",
        local_time: "11:00 PM - 12:00 AM JST",
        explanation: "This is very late for Charlie, making it less ideal."
      },
      {
        option: "12:00 PM - 1:00 PM GMT",
        local_time: "9:00 PM - 10:00 PM JST",
        explanation: "This is the most convenient time for Charlie as it's earlier in the evening."
      }
    ]
  }
];

export const MemberList = ({ data, total }: MemberListProps) => {
  const workspaceId = useWorkspaceId();

  return (
    <div className="my-8">
      <div className="bg-white border rounded-lg p-4">
        <div className="flex items-center justify-between gap-10">
          <p className="text-lg font-semibold">
            Members ({total})
          </p>
          <Button asChild variant="secondary" size="icon">
            <Link href={`/workspaces/${workspaceId}/members`}>
              <SettingsIcon className="size-4 text-neutral-400" />
            </Link>
          </Button>
        </div>
        <DottedSeparator className="my-4" />
        <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.map((member) => (
            <li key={member.$id}>
              <Card className="shadow-none rounded-lg overflow-hidden">
                <CardContent className="p-3 flex flex-col items-center gap-x-2">
                  <MemberAvatar name={member.name} className="size-12" />
                  <div className="flex flex-col items-center overflow-hidden">
                    <p className="text-lg font-medium line-clamp-1">
                      {member.name}
                    </p>
                    <p className="text-sm text-muted-foreground line-clamp-1">
                      {member.email}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </li>
          ))}
          <li className="text-sm text-muted-foreground text-center hidden first-of-type:block">
            No members found
          </li>
        </ul>
      </div>

      <div className="border border-[#E4E7EB] my-8">
      <p className="mt-4 mx-4 font-semibold text-xl">AI Suggestions</p>
      <div className="grid grid-cols-3">
        {membersData.map((member, index) => (
          <div key={index} className="space-y-4 ">
            <div className="m-4 p-4 mt-10 border border-[#E4E7EB] ">
              <p className="text-lg font-semibold">Name: {member.name}</p>
              <p>Location: {member.location}</p>
              <p>Timezone: {member.timezone}</p>
              <p>Working Hours: {member.working_hours}</p>
              <p>GMT Offset: {member.gmt_offset}</p>
            </div>
          </div>
        ))}
      </div>
      </div>
    </div>
  );
};
