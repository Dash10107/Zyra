import DottedSeparator from "@/components/dotted-separator"
import Image from "next/image"
import Link from "next/link"
import { Navigation } from "./Navigation"
import { WorkspaceSwitcher } from "./WorkspaceSwitcher"
import { Projects } from './projects';


export const Sidebar = () => {
  return (
    <aside className="h-full bg-neutral-100 p-4 w-full">
      <Link href="/">
        <p className="font-extrabold text-3xl text-[#2463eb]">ZYRA</p>
      </Link>
      <DottedSeparator className="my-4"/>
      <WorkspaceSwitcher />
      <DottedSeparator className="my-4" />
      <Navigation />
      <DottedSeparator className="my-4" />
      <Projects />

    </aside>
  )
}