import { FastifyInstance, FastifyRequest } from 'fastify';
import config from '../config';

interface TokenType {
    userId: string;
    username: string;
    isAdmin: string;
    iat: string;
}

const extractIdFromRequest = (req: FastifyRequest): string | undefined => {
    const decodedToken:TokenType | null = req.server.jwt.decode(req.appToken.get('data') || '');
    return decodedToken?.userId;
}

export default extractIdFromRequest;