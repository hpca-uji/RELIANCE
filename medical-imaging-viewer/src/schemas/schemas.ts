import { FastifyReply, FastifyRequest } from "fastify";
import { postContact, postLabels, postDiscard, postLogin } from "../controllers/controllers";

export const loginOpts = {
    schema: {
        body: {
            type: 'object',
            properties: {
                username: {type: 'string'},
                password: {type: 'string'},
            }
        },
        response: {
            200: {
                type: 'object',
                properties: {
                    data: {
                        type: 'object',
                        properties: {
                            userId: {type: 'integer'},
                            username: {type: 'string'},
                            email: {type: 'string'},
                            isAdmin: {type: 'boolean'},
                        }
                    },
                    token: {type: 'string'},
                }
            },
            401: {
                error: {type: 'string'},
            }
        }
    },
    preValidation: async (request: FastifyRequest<{ Body: {username: string, password: string} }>, reply: FastifyReply) => {
        const { username, password } = request.body;

        if ((typeof username === 'undefined' || !username)
                || (typeof password === 'undefined' || !password)) {
                    return reply.view('login.ejs', { error: "Username or password was not provided." })
                }
        return;
    },
    handler: postLogin
}

export const postContactOpts = {
    schema: {
        body: {
            type: 'object',
            properties: {
                username: {type: 'string'},
                email: {type: 'string'},
                subject: {type: 'string'},
            }
        },
        response: {
            200: {
                msg: {type: 'string'},
            },
            401: {
                error: {type: 'string'},
            }
        }
    },
    preValidation: async (request: FastifyRequest<{ Body: {username: string, email: string, subject: string} }>, reply: FastifyReply) => {
        const { username, email, subject } = request.body;

        if ((typeof username === 'undefined' || !username)
                || (typeof email === 'undefined' || !email)
                || (typeof subject === 'undefined' || !subject)) {
                    return reply.view('contact.ejs', { error: "Username, email address, or message was not provided." });
                }
        return;
    },
    handler: postContact
}

export const postLabelsOpts = {
    schema: {
        body: {
            checkedLabels: {
                type: 'array',
                items: { type: 'string' },
            },
            studyInstanceUID: {
                type: 'string',
            },
            seriesInstanceUID: {
                type: 'string',
            },
            sopInstanceUID: {
                type: 'string',
            }
        },
        response: {
            200: {
                msg: {type: 'string'},
            },
            401: {
                error: {type: 'string'},
            }
        }
    },
    preValidation: async (request: FastifyRequest<{ Body: {checkedLabels: string[], studyInstanceUID: string, seriesInstanceUID: string, sopInstanceUID: string} }>, reply: FastifyReply) => {
        const { studyInstanceUID, seriesInstanceUID, sopInstanceUID } = request.body;

        if ((typeof studyInstanceUID === 'undefined' || !studyInstanceUID)
                || (typeof seriesInstanceUID === 'undefined' || !seriesInstanceUID)
                || (typeof sopInstanceUID === 'undefined' || !sopInstanceUID)) {
                    return reply.code(401).send('Some body parameters might be missing.');
                }
        return;
    },
    handler: postLabels
}

export const postDiscardOpts = {
    schema: {
        body: {
            studyInstanceUID: {
                type: 'string',
            },
            seriesInstanceUID: {
                type: 'string',
            },
            sopInstanceUID: {
                type: 'string',
            }
        },
        response: {
            200: {
                msg: {type: 'string'},
            },
            401: {
                error: {type: 'string'},
            }
        }
    },
    preValidation: async (request: FastifyRequest<{ Body: {studyInstanceUID: string, seriesInstanceUID: string, sopInstanceUID: string} }>, reply: FastifyReply) => {
        const { studyInstanceUID, seriesInstanceUID, sopInstanceUID } = request.body;

        if ((typeof studyInstanceUID === 'undefined' || !studyInstanceUID)
                || (typeof seriesInstanceUID === 'undefined' || !seriesInstanceUID)
                || (typeof sopInstanceUID === 'undefined' || !sopInstanceUID)) {
                    return reply.code(401).send('Some body parameters might be missing.');
                }
        return;
    },
    handler: postDiscard
}
