

import { redirect } from 'next/navigation';
import { getWorkspaces } from '@/features/workspaces/queries';




const Home = async() => {

  const workspaces = await getWorkspaces();
  if(workspaces.total === 0) {
    redirect("/dashboard/workspaces/create")
  }else{
    redirect(`/dashboard/workspaces/${workspaces.documents[0].$id}`)
  }

  
}

export default Home;