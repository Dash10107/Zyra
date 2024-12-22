

import { redirect } from 'next/navigation';
import { getCurrent } from '@/features/auth/queries';
import { getWorkspaces } from '@/features/workspaces/queries';




const Home = async() => {

  const user = await getCurrent();
  // if(!user) redirect("/sign-in")

  const workspaces = await getWorkspaces();
  if(workspaces.total === 0) {
    redirect("/dashboard/workspaces/create")
  }else{
    redirect(`/dashboard/workspaces/${workspaces.documents[0].$id}`)
  }

  
}

export default Home;