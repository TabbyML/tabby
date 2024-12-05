import { UserGroupsQuery } from '@/lib/gql/generates/graphql'

export type MemberShips = UserGroupsQuery['userGroups'][0]['members']
export type MemberShipUser = MemberShips[number]['user']
